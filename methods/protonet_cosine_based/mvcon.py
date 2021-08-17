import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from methods.protonet_cosine import ProtoNet_cosine
from methods import backbone
import ipdb


class UFSL(ProtoNet_cosine):
  maml = False

  def __init__(self, model_func, tau, pos_wt_type, n_way, n_support=5, n_support_u=1, tf_path=None, loadpath=None,
               is_distribute=False):
    super(UFSL, self).__init__(model_func, n_way, n_support=n_support, tf_path=tf_path)

    self.tau = tau
    self.pos_wt_type = pos_wt_type

    self.projected_feature_dim = 128
    self.projection_head = nn.Sequential(
      # nn.Linear(self.feature.final_feat_dim, self.feature.final_feat_dim),
      # nn.ReLU(),
      nn.Linear(self.feature.final_feat_dim, self.projected_feature_dim)
    )
    # self.projection_head = None

    self.method = 'UFSL'
    self.n_support_u = n_support_u
    if loadpath != None:
      self.load_model(loadpath)
    if is_distribute:
      self.distribute_model1()

  def load_model(self, loadpath):
    state = torch.load(loadpath)['state']
    loadstate = {}
    for key in state.keys():
      if '.module' in key:
        newkey = key.replace('.module', '')
      else:
        newkey = key
      loadstate[newkey] = state[key]
    self.load_state_dict(loadstate, strict=False)
    return self

  def resume_best_ufsl(self, loadpath):
    loadfile = torch.load(loadpath)
    state = loadfile['state']
    self.load_state_dict(state, strict=False)
    resume_epoch = loadfile['epoch']
    return resume_epoch

  def distribute_model1(self):
      self.feature = nn.DataParallel(self.feature)
      return self

  def forward_this(self, x):
    r = self.feature(x)
    if self.projection_head is not None:
      z = self.projection_head(r)
    else:
      z = r
    return z, r

  def set_forward_loss_target(self, x, n_way, n_support, n_query):
    y_query = torch.from_numpy(np.repeat(range(n_way), n_query))
    y_query = y_query.cuda()
    x = x.cuda()
    x = x.contiguous().view(n_way * (n_support + n_query), *x.size()[2:])
    _, z_all = self.forward_this(x)
    # z_all = F.normalize(z_all, dim=1)
    z_all = z_all.view(n_way, n_support + n_query, -1)
    z_support = z_all[:, :n_support]
    z_query = z_all[:, n_support:]
    z_support = z_support.contiguous()
    z_proto = z_support.view(n_way, n_support, -1).mean(1)  # the shape of z is [n_data, n_dim]
    z_query = z_query.contiguous().view(n_way * n_query, -1)
    scores = cosine_dist(z_query, z_proto)
    loss = self.loss_fn(scores, y_query)
    return scores, loss

  def _mvcon_loss(self, z, n_way, n_s):
    # normalize
    z = F.normalize(z, dim=1)
    bsz, featdim = z.size()
    z_square = z.view(bsz, 1, featdim).repeat(1, bsz, 1)
    # similarity of vectorized nway kshot task
    Sv = nn.CosineSimilarity(dim=2)(z_square, z_square.transpose(1, 0))
    Sv = torch.exp(Sv / self.tau)
    denominator = (Sv - Sv.diag().diag()).sum(dim=1).unsqueeze(1)
    # create appropriate mask to compute the numerator and denominator
    mask = torch.FloatTensor(np.kron(np.eye(n_way), np.ones((n_s, n_s)))).to(Sv.device)
    if self.pos_wt_type == 'uniform':
      Sv_norm = Sv / denominator
      T1 = torch.log(Sv_norm)
      # self loss, s_ii is omitted
      T1 = T1 - T1.diag().diag()
      T1_pos = mask * T1
      li = (1.0 / (n_s - 1)) * T1_pos.sum(dim=1)
    elif self.pos_wt_type == 'min':
      S = mask * Sv + (1 - mask) * Sv.max()
      numerator = S.min(dim=1)[0]
      li = torch.log(numerator / denominator)
    # loss = -self.tau * li.sum(dim=0)
    loss = -li.mean(dim=0)
    return loss

  def mvcon_loss(self, x, mode='train'):
    n_way = x.size(0)
    n_s = x.size(1)
    x = x.cuda()
    x = x.contiguous().view(n_way * n_s, *x.size()[2:])
    z, r = self.forward_this(x)
    if mode == 'train':
      return self._mvcon_loss(z, n_way, n_s)
    elif mode == 'val':
      lp = self._mvcon_loss(z, n_way, n_s)
      return lp

  def train_target(self, epoch, tgt_loader, optimizer):
    print_freq = 10
    for i, (x_tgt, y_tgt) in enumerate(tgt_loader):
      loss = self.mvcon_loss(x_tgt, 'train')
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      if i % print_freq == 0:
        bch_id = epoch * len(tgt_loader) + i
        print('Epoch {:d} | Batch {:d} | Loss {:f}'.format(epoch, i, loss.item()))
        self.tf_writer.add_scalar(self.method + '/mvcon_loss/total', loss.item(), bch_id)
    return

  def validate(self, epoch, test_loader):
    loss_projhead = 0.
    for i, (x, y) in enumerate(test_loader):
      loss_this_projhead  = self.mvcon_loss(x, 'val')
      loss_projhead += loss_this_projhead.item()
    loss_projhead = loss_projhead / len(test_loader)
    print('--- EPOCH %d, Loss_projhead = %.6f ---' % (epoch, loss_projhead))
    return loss_projhead, loss_projhead

  def test_unsupervised(self, test_loader):
    loss = 0.
    count = 0
    acc_all = []
    iter_num = len(test_loader)
    for i, (x, y) in enumerate(test_loader):
      n_query_u = x.size(1) - self.n_support_u
      if self.change_way:
        n_way = x.size(0)
      scores, loss = self.set_forward_loss_target(x, n_way, self.n_support_u, n_query_u)
      y_query = np.repeat(range(n_way), n_query_u)
      topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
      topk_ind = topk_labels.cpu().numpy()
      top1_correct = np.sum(topk_ind[:, 0] == y_query)
      correct_this = float(top1_correct)
      count_this = len(y_query)
      loss_this = loss.item() * len(y_query)
      acc_all.append(correct_this / count_this * 100)
      loss += loss_this
      count += count_this
    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print('--- %d Loss = %.6f ---' % (iter_num, loss / count))
    print('--- %d Test Acc = %4.2f%% +- %4.2f%% ---' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
    return acc_mean


  def test_supervised(self, test_loader, record = None):
    loss = 0.
    count = 0
    acc_all = []
    iter_num = len(test_loader)
    for i, (x,y) in enumerate(test_loader):
      n_query = x.size(1) - self.n_support
      if self.change_way:
        n_way  = x.size(0)
      scores, loss = self.set_forward_loss_target(x, n_way, self.n_support, n_query)
      y_query = np.repeat(range(n_way), n_query)
      topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
      topk_ind = topk_labels.cpu().numpy()
      top1_correct = np.sum(topk_ind[:, 0] == y_query)
      correct_this = float(top1_correct)
      count_this = len(y_query)
      loss_this = loss.item()*len(y_query)
      acc_all.append(correct_this/ count_this*100)
      loss += loss_this
      count += count_this
    acc_all  = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = 1.96*np.std(acc_all)/np.sqrt(iter_num)
    print('--- %d Loss = %.6f ---' %(iter_num,  loss/count))
    print('--- %d Test Acc = %4.2f%% +- %4.2f%% ---' %(iter_num,  acc_mean, acc_std))
    return acc_mean, acc_std

def cosine_dist(x, y):
  # x: N x D
  # y: M x D
  n = x.size(0)
  m = y.size(0)
  d = x.size(1)
  assert d == y.size(1)

  x = x.unsqueeze(1).expand(n, m, d)
  y = y.unsqueeze(0).expand(n, m, d)
  alignment = nn.functional.cosine_similarity(x, y, dim=2)
  return alignment
