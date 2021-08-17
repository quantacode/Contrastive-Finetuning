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
import random
import ipdb

class HPM_model(SIMCLR):
  def __init__(self, model_func, tau, beta, image_size, n_way, tf_path=None, loadpath=None, is_distribute=False):
    super(HPM_model, self).__init__(model_func, tau, n_way, tf_path=tf_path)
    self.method = 'Npcon_Model'
    self.beta=beta
    self.image_size=image_size
    self.load_models(loadpath)
    if is_distribute:
      self.distribute_model1()

  def load_models(self, loadpath):
    if loadpath is not None:
      state = torch.load(loadpath)['state']
      loadstate = {}
      for key in state.keys():
        if 'projection_head' not in key:
          newkey = key
          if '.module' in key:
            newkey = key.replace('.module', '')
          loadstate[newkey] = state[key]
      self.load_state_dict(loadstate, strict=False)
    return self

  def distribute_model1(self):
      self.feature = nn.DataParallel(self.feature)
      return self

  def _hpm_loss(self, z, mask, n_s=2):
    # simclr
    z = F.normalize(z, dim=1)
    bsz, featdim = z.size()
    z_square = z.view(bsz, 1, featdim).repeat(1, bsz, 1)
    sim = nn.CosineSimilarity(dim=2)(z_square, z_square.transpose(1, 0))
    Sv = torch.exp(sim / self.tau)
    neg = (Sv * (1 - mask)).sum(dim=1).unsqueeze(1)
    li = mask * torch.log(Sv / (Sv + neg) + 0.000001)
    li = li - li.diag().diag()
    li = (1 / (n_s - 1)) * li.sum(dim=1)
    loss_simclr = -li.mean(dim=0)

    # hpm_reg
    ## original formulation
    # T = torch.log(Sv/(Sv+neg))
    # modified
    T = torch.log(Sv/neg)
    weight = torch.exp(-self.beta * sim) * (1 - mask)
    weight = weight/weight.sum(dim=1).unsqueeze(1)
    reg_i = (weight*T).sum(dim=1)
    loss_hpm = reg_i.mean(dim=0)

    return loss_simclr, loss_hpm, sim

  def hpm_loss(self, x):
    n_way = x.size(0)
    n_s = x.size(1)
    x = x.cuda()
    x = x.contiguous().view(n_way * n_s, *x.size()[2:])
    z, r = self.forward_this(x)
    mask = torch.FloatTensor(np.kron(np.eye(n_way), np.ones((n_s, n_s)))).to(z.device)
    return self._hpm_loss(z, mask)


  def train_simclr_hpm(self, epoch, tgt_loader, optimizer, reg_wt=1.0):
    # train npcon based on source-model metric
    progress = tqdm(tgt_loader)
    for i, (x, y) in enumerate(progress):
      loss_simclr, hpm_reg, sim = self.hpm_loss(x)
      loss = loss_simclr + reg_wt * hpm_reg
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      bch_id = epoch * len(tgt_loader) + i
      self.tf_writer.add_scalar(self.method + '/simclr_hpm_loss/simclr', loss_simclr.item(), bch_id)
      self.tf_writer.add_scalar(self.method + '/simclr_hpm_loss/hpm_reg', hpm_reg.item(), bch_id)
      self.tf_writer.add_scalar(self.method + '/simclr_hpm_loss/total', loss.item(), bch_id)
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