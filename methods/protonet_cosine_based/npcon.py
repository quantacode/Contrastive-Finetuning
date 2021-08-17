# nearest positve contrastive learning
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from methods.protonet_cosine_based.simclr import SIMCLR
from methods import backbone
from data.datamgr import NamedDataManager
from tqdm import tqdm
from scipy import stats
import random
import ipdb

class Npcon_Model(SIMCLR):
  def __init__(self, model_func, tau, affinity_file, image_size, naug, n_way, tf_path=None, loadpath=None, is_distribute=False):
    super(Npcon_Model, self).__init__(model_func, tau, n_way, tf_path=tf_path)
    self.method = 'Npcon_Model'
    self.affinity_file = affinity_file
    self.image_size=image_size
    self.naug = naug
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

  def distribute_model1(self):
      self.feature = nn.DataParallel(self.feature)
      return self

  def np_sampling_accuracy(self, affinity_graph, data, ns, mode, epoch):
    # compute prob of majority samples belonging to same class
    indices = np.arange(len(data['image_label']))
    random.shuffle(indices)
    uniqueL = np.unique(data['image_label'])
    prob = {}
    for uL in uniqueL:
      prob[uL] = []
    for idx in indices:
      closest_idx = np.argsort(affinity_graph[idx])
      anchor_class = data['image_label'][idx]
      closest_classes = [anchor_class]
      for cid in closest_idx[-(ns - 1):]:
        closest_classes.append(data['image_label'][cid])
      prob[anchor_class].append(stats.mode(closest_classes)[1].item() >= mode)
    precision=np.mean([np.mean(val) for key, val in prob.items()])
    self.tf_writer.add_scalar('parameters/nearpos_success_precision', precision, epoch)
    ipdb.set_trace()
    return precision, mode, ns

  def group_cosine_dist(self, features):
    bsz = features.size(0)
    grpsz = 200
    grps = int(np.ceil(bsz/grpsz))
    affinity_matrix = torch.zeros(bsz,bsz)
    progress = tqdm(range(grps))
    progress.set_description('computing affinity')
    for g1 in progress:
      start1 = g1*grpsz
      end1 = np.clip((g1+1)*grpsz, 0, bsz)
      feat1 = features[start1:end1]
      for g2 in range(g1, grps):
        start2 = g2 * grpsz
        end2 = np.clip((g2 + 1) * grpsz, 0, bsz)
        feat2 = features[start2:end2]
        cdist = cosine_dist(feat1, feat2)
        affinity_matrix[start1:end1, start2:end2] = cdist
        affinity_matrix[start2:end2, start1:end1] = cdist.transpose(0,1)
    affinity_matrix = affinity_matrix - affinity_matrix.diag().diag()
    return affinity_matrix.cpu().numpy()

  def get_affinity(self, epoch):
    self.eval()
    datamgr = NamedDataManager(self.image_size, batch_size=64, is_shuffle=False)
    data_loader = datamgr.get_data_loader(self.affinity_file, aug=False)
    data = {'image_name': [], 'image_feat': torch.FloatTensor([]).cuda(), 'image_label': [], 'cluster_label': []}
    progress = tqdm(data_loader)
    progress.set_description('generating features')
    for i, (name, x, y) in enumerate(progress):
      x = x.cuda()
      feat = self.forward_this(x)[1]
      feat = F.normalize(feat, dim=1)
      feat = feat.detach()
      y = list(y.numpy())
      data['image_name'].extend(name)
      data['image_feat'] = torch.cat([data['image_feat'], feat], dim=0)
      data['image_label'].extend(y)
    features = data['image_feat']
    affinity_matrix = self.group_cosine_dist(features)
    prob, mode, ns = self.np_sampling_accuracy(affinity_matrix, data,
                               ns=self.naug, mode=np.max([int(np.ceil(self.naug / 2)), 2]), epoch=epoch)
    progress.refresh()
    progress.set_description("Probability of meaninful cluster (%d/%d)= %0.2f" % (mode, ns, prob))
    return affinity_matrix

  def train_target_npcon(self, epoch, tgt_con_loader, tgt_loader, optimizer, nploss_wt):
    assert len(tgt_loader) == len(tgt_con_loader)
    progress = tqdm(range(len(tgt_loader)))
    tgt_loader = iter(tgt_loader)
    tgt_con_loader = iter(tgt_con_loader)
    for i in progress:
      pos1, pos2, _ = next(tgt_con_loader)
      x_con = torch.cat([pos1.unsqueeze(1), pos2.unsqueeze(1)], dim=1)
      loss_con = self.simclr_loss(x_con)

      x, _ = next(tgt_loader)
      loss_npcon = self.simclr_loss(x)

      loss = loss_con + nploss_wt * loss_npcon
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      bch_id = epoch * len(tgt_loader) + i
      self.tf_writer.add_scalar(self.method + '/npcon_loss/contrastive', loss_con.item(), bch_id)
      self.tf_writer.add_scalar(self.method + '/npcon_loss/npcon', loss_npcon.item(), bch_id)
      self.tf_writer.add_scalar(self.method + '/npcon_loss/total', loss.item(), bch_id)
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
