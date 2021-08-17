# nearest positve contrastive learning
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from methods.protonet_cosine_based.simclr import SIMCLR
from methods import backbone
from data.nearest_positives_datamgr import NamedDataManager
# from data.datamgr import NamedDataManager
from tqdm import tqdm
from scipy import stats
from methods.protonet_cosine_based.HPM_utils import get_hp_indices
import random
import ipdb

class Npcon_Model(SIMCLR):
  def __init__(self, model_func, tau, affinity_file, image_size, naug, hpm_type, num_classes, n_way,
               tf_path=None, loadpath=None, is_distribute=False):
    super(Npcon_Model, self).__init__(model_func, tau, n_way, tf_path=tf_path)
    self.method = 'Baseline_NPcon'
    self.affinity_file = affinity_file
    self.image_size=image_size
    self.naug = naug
    self.hpm_type = hpm_type
    self.classifier = nn.Linear(self.feature.final_feat_dim, num_classes)
    self.load_models(loadpath)
    if is_distribute:
      self.distribute_model1()

  def load_models(self, loadpath):
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

  def distribute_model1(self):
      self.feature = nn.DataParallel(self.feature)
      return self

  def np_sampling_accuracy_CUB(self, affinity_graph, data, ns, mode, epoch):
    # compute prob of majority samples belonging to same class
    indices = np.arange(len(data['image_label']))
    random.shuffle(indices)
    npos = ns-1
    assert npos==1
    prob = []
    for idx in indices:
      closest_classes = [data['image_name'][idx].split('/')[-2].split('.')[1].split('_')[-1]]
      closest_idx = get_hp_indices(affinity_graph[idx], npos, self.hpm_type)
      for cid in closest_idx[-npos:]:
        closest_classes.append(data['image_name'][cid].split('/')[-2].split('.')[1].split('_')[-1])
      prob.append(stats.mode(closest_classes)[1].item() >= mode)
    self.tf_writer.add_scalar('parameters/interp_success_prob', np.mean(prob), epoch)
    return np.mean(prob), mode, ns

  # def np_sampling_accuracy_CUB(self, affinity_graph, data, ns, mode, epoch):
  #   # compute prob of majority samples belonging to same class
  #   indices = np.arange(len(data['image_label']))
  #   random.shuffle(indices)
  #   prob = []
  #   for idx in indices:
  #     closest_classes = [data['image_name'][idx].split('/')[-2].split('.')[1].split('_')[-1]]
  #     closest_idx = np.argsort(affinity_graph[idx])
  #     for cid in closest_idx[-(ns - 1):]:
  #       closest_classes.append(data['image_name'][cid].split('/')[-2].split('.')[1].split('_')[-1])
  #     prob.append(stats.mode(closest_classes)[1].item() >= mode)
  #   self.tf_writer.add_scalar('parameters/interp_success_prob', np.mean(prob), epoch)
  #   return np.mean(prob), mode, ns

  def np_sampling_accuracy_CUB2(self, affinity_graph, data, k_nearest, ns=2, mode=2, epoch=-1):
    # compute prob of majority samples belonging to same class
    indices = np.arange(len(data['image_label']))
    random.shuffle(indices)
    prob = []
    for idx in indices:
      cid = stats.mode(torch.argsort(affinity_graph[idx],dim=0)[-k_nearest:,:].view(-1).numpy())[0][0]
      closest_classes = [data['image_name'][idx].split('/')[-2].split('.')[1].split('_')[-1]]
      closest_classes.append(data['image_name'][cid].split('/')[-2].split('.')[1].split('_')[-1])
      prob.append(stats.mode(closest_classes)[1].item() >= mode)
    self.tf_writer.add_scalar('parameters/interp_success_prob', np.mean(prob), epoch)
    return np.mean(prob), mode, ns

  def np_sampling_accuracy(self, affinity_graph, data, ns, mode, epoch):
    # compute prob of majority samples belonging to same class
    indices = np.arange(len(data['image_label']))
    random.shuffle(indices)
    uniqueL = np.unique(data['image_label'])
    prob = {}
    for uL in uniqueL:
      prob[uL] = []
    for idx in indices:
      closest_idx = torch.argsort(affinity_graph[idx], axis=0)
      anchor_class = data['image_label'][idx]
      closest_classes = [anchor_class]
      for cid in closest_idx[-(ns - 1):]:
        closest_classes.append(data['image_label'][cid])
      prob[anchor_class].append(stats.mode(closest_classes)[1].item() >= mode)
    precision=np.mean([np.mean(val) for key, val in prob.items()])
    self.tf_writer.add_scalar('parameters/nearpos_success_precision', precision, epoch)
    return precision, mode, ns

  def group_dist(self, features, tau_aff):
    bsz = features.size(0)
    grpsz = 200
    grps = int(np.ceil(bsz / grpsz))
    affinity_matrix = torch.zeros(bsz, bsz)
    for g1 in range(grps):
      start1 = g1 * grpsz
      end1 = np.clip((g1 + 1) * grpsz, 0, bsz)
      feat1 = features[start1:end1]
      for g2 in range(g1, grps):
        start2 = g2 * grpsz
        end2 = np.clip((g2 + 1) * grpsz, 0, bsz)
        feat2 = features[start2:end2]
        cdist = cosine_dist(feat1, feat2)
        # cdist = euclidean_dist(feat1, feat2)
        affinity_matrix[start1:end1, start2:end2] = cdist
        affinity_matrix[start2:end2, start1:end1] = cdist.transpose(0, 1)
    affinity_matrix = torch.exp(affinity_matrix / tau_aff)
    # affinity_matrix = torch.exp(-affinity_matrix/tau_aff)
    affinity_matrix = affinity_matrix - affinity_matrix.diag().diag()
    return affinity_matrix.cpu()

  def get_affinity(self, epoch):
    # multiple model based affinity
    self.eval()
    batch_size = 8
    num_aug_aff = 20
    datamgr = NamedDataManager(self.image_size, num_aug=num_aug_aff, batch_size=batch_size, is_shuffle=False)
    data_loader = datamgr.get_data_loader(self.affinity_file, aug=True)
    data = {'image_name': [], 'image_feat': torch.FloatTensor([]).cuda(), 'image_label': [], 'cluster_label': []}
    progress = tqdm(data_loader)
    progress.set_description('generating features')
    for i, (name, x, y) in enumerate(progress):
      data['image_name'].extend(name)
      y = list(y.numpy())
      data['image_label'].extend(y)
      x = x.cuda()
      bsz, views = x.size(0), x.size(1)
      x = x.contiguous().view(bsz * views, *x.size()[2:])
      feat = self.feature(x).detach()
      feat = F.normalize(feat, dim=1)
      featdim = feat.size(-1)
      feat = feat.view(bsz, views, featdim)
      data['image_feat'] = torch.cat([data['image_feat'], feat], dim=0)
    tau_aff = 0.1
    affinity_list = [self.group_dist(feat.squeeze(1), tau_aff) for feat in data['image_feat'].split(1, dim=1)]
    affinity_list = [(aff / (aff.sum(dim=1).unsqueeze(1))).unsqueeze(2) for aff in affinity_list]
    affinity_matrix = torch.cat(affinity_list, dim=2)

    mode = np.max([int(np.ceil(self.naug / 2)), 2])
    if 'CUB' in self.affinity_file:
      prob, mode, ns = self.np_sampling_accuracy_CUB(affinity_matrix, data, ns=self.naug, mode=mode, epoch=epoch)
    else:
      prob, mode, ns = self.np_sampling_accuracy(affinity_matrix, data, ns=self.naug, mode=mode, epoch=epoch)
    # print(mode, ns, "%0.4f"%(prob))
    # exit()
    return affinity_matrix

  def _kNearest_loss(self, z, mask_pos, mask_neg, n_s):
    # normalize
    z = F.normalize(z, dim=1)
    bsz, featdim = z.size()
    z_square = z.view(bsz, 1, featdim).repeat(1, bsz, 1)
    Sv = nn.CosineSimilarity(dim=2)(z_square, z_square.transpose(1, 0))
    Sv = torch.exp(Sv / self.tau)
    neg = (Sv*mask_neg).sum(dim=1).unsqueeze(1).repeat(1,bsz)
    li = mask_pos*torch.log(Sv/(Sv+neg)+0.000001)
    li = li-li.diag().diag()
    li = (1/(n_s-1))*li.sum(dim=1)
    # Note: dividing li.sum by mask.sum takes care for different strategies of incuding paired-positives in the loss
    # in the vanila case (simclr), the operation boils down to li.mean
    active_anchors = mask_pos.sum(dim=1)>0
    loss = -li.sum(dim=0)/active_anchors.sum(dim=0)
    return loss

  def kNearest_mvclr(self, x):
    n_way,n_s = x.size()[:2]
    x = x.cuda()
    x = x.contiguous().view(n_way * n_s, *x.size()[2:])
    z, _ = self.forward_this(x)
    T1 = np.eye(n_way)
    T2 = np.vstack([np.ones((1, n_s)), np.zeros((n_s-1,n_s))])
    mask_pos = torch.FloatTensor(np.kron(T1, T2)).to(z.device)
    # return self._simclr_loss(z, mask_pos, n_s)
    T1 = np.ones((n_way, n_way))
    T1 = T1 - np.diag(np.diag(T1))
    # T2 = np.zeros((n_s, n_s))
    # T2[0,0]=1
    T2 = np.ones((n_s, n_s))
    mask_neg = torch.FloatTensor(np.kron(T1, T2)).to(z.device)
    return self._kNearest_loss(z, mask_pos, mask_neg, n_s)

  def baseline_loss(self, x, y):
    x, y = x.cuda(), y.cuda()
    _, r = self.forward_this(x)
    scores = self.classifier(r)
    y = y.cuda()
    return self.loss_fn(scores, y)

  def train_BSNPT(self, epoch, src_loader, tgt_loader, optimizer, loss_wt, mode='nearpos'):
    # Baseline-Src NPcon-Tgt
    if mode == 'nearpos':
      tgt_forward_func = self.kNearest_mvclr
    else:
      tgt_forward_func = self.simclr_loss
    assert len(src_loader) % len(tgt_loader) == 0
    src_iter = iter(src_loader)
    tgt_iter = iter(tgt_loader)
    progress = tqdm(range(len(src_loader)))
    for i in progress:
      # source
      x_src, y_src = next(src_iter)
      loss_src = self.baseline_loss(x_src, y_src)

      # target
      if (i+1)%len(tgt_loader)==0:
        tgt_iter = iter(tgt_loader)
      x_tgt, y_tgt = next(tgt_iter)
      loss_tgt = tgt_forward_func(x_tgt)

      loss = loss_src + loss_wt * loss_tgt
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      bch_id = epoch * len(src_loader) + i
      self.tf_writer.add_scalar(self.method + '/source_loss', loss_src.item(), bch_id)
      self.tf_writer.add_scalar(self.method + '/target_loss', loss_tgt.item(), bch_id)
      self.tf_writer.add_scalar(self.method + '/TOTAL_loss', loss.item(), bch_id)
      progress.set_description('Epoch {:d} | Batch {:d} | Loss {:f}'.format(epoch, i, loss.item()))
    return

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

