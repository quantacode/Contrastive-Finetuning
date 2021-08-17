import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from methods.protonet_cosine_based.mvcon import UFSL as MVCON
from methods import backbone
import ipdb


class UFSL(MVCON):
  maml = False

  def __init__(self, model_func, tau, tau_l1, pos_wt_type, n_way, n_support, n_support_u=1, tf_path=None,
               loadpath=None):
    super(UFSL, self).__init__(model_func,
                               tau=tau,
                               pos_wt_type=pos_wt_type, 
                               n_way=n_way, n_support=n_support, n_support_u=n_support_u, 
                               tf_path=tf_path,
                               loadpath=loadpath)
    self.tau_l1 = tau_l1
    self.method = 'UFSL'

  def sparsity_regularizer(self, z):
    bsz = z.size(0)
    adjacency = torch.sigmoid(cosine_dist(z, z)/self.tau_l1)
    adjacency = torch.triu(adjacency, diagonal=1)
    mask = torch.tril(-torch.ones(adjacency.size())).to(adjacency.device)
    adjacency += mask
    # loss = adjacency[idx].mean()
    loss = adjacency[adjacency!=-1].sum()/((bsz**2 - bsz)/2)
    return loss

  def mvcon_loss(self, x, mode='train'):
    n_way = x.size(0)
    n_s = x.size(1)
    x = x.cuda()
    x = x.contiguous().view(n_way * n_s, *x.size()[2:])
    z, r = self.forward_this(x)
    if mode == 'train':
      return self._mvcon_loss(z, n_way, n_s), self.sparsity_regularizer(z)
    elif mode == 'val':
      lp = self._mvcon_loss(z, n_way, n_s)
      return lp

  def train_target(self, epoch, tgt_loader, optimizer, reg_wt=1.0):
    print_freq = 10
    for i, (x_tgt, y_tgt) in enumerate(tgt_loader):
      self.n_way = x_tgt.size(0)
      loss_primary, reg_loss = self.mvcon_loss(x_tgt, 'train')
      loss = loss_primary + reg_wt*reg_loss

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      if i % print_freq == 0:
        bch_id = epoch * len(tgt_loader) + i
        print('Epoch {:d} | Batch {:d} | Loss {:f}'.format(epoch, i, loss.item()))
        self.tf_writer.add_scalar(self.method + '/mvcon_loss/primary', loss_primary.item(), bch_id)
        self.tf_writer.add_scalar(self.method + '/mvcon_loss/reg', reg_loss.item(), bch_id)
        self.tf_writer.add_scalar(self.method + '/mvcon_loss/total', loss.item(), bch_id)
    return

  def validate(self, epoch, test_loader):
    loss_projhead = 0.
    for i, (x, y) in enumerate(test_loader):
      loss_this_projhead  = self.mvcon_loss(x, 'val')
      loss_projhead += loss_this_projhead.item()
    loss_projhead = loss_projhead / len(test_loader)
    print('--- %d Loss_projhead = %.6f ---' % (epoch, loss_projhead))
    return loss_projhead, loss_projhead

  def test_unsupervised(self, test_loader):
    loss = 0.
    count = 0
    acc_all = []
    iter_num = len(test_loader)
    for i, (x, y) in enumerate(test_loader):
      self.n_query_u = x.size(1) - self.n_support_u
      if self.change_way:
        self.n_way = x.size(0)
      scores, loss = self.set_forward_loss_target(x, self.n_way, self.n_support_u, self.n_query_u)
      y_query = np.repeat(range(self.n_way), self.n_query_u)
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
      self.n_query = x.size(1) - self.n_support
      if self.change_way:
        self.n_way  = x.size(0)
      scores, loss = self.set_forward_loss_target(x, self.n_way, self.n_support, self.n_query)
      y_query = np.repeat(range(self.n_way), self.n_query)
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
