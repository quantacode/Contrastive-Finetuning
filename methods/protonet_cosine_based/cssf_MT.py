# contrastive semi supervised model
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
from methods.protonet_cosine_based.cssf import CSSF
import random
import ipdb
eps=0.000001
class CSSF_MT(CSSF):
  def __init__(self, model_func, tau, n_way, tf_path, loadpath, is_distribute, is_same_head=True,
               beta=0, clstau=0, n_trim=1):
    super(CSSF_MT, self).__init__(model_func,tau,n_way,tf_path,beta=beta,clstau=clstau,n_trim=n_trim)
    self.method = 'CSSF_MT'
    self.is_same_head = is_same_head
    if not is_same_head:
      self.projection_head_LPUN = nn.Sequential(
        nn.Linear(self.feature.final_feat_dim, self.projected_feature_dim, bias=True)
      )
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

      if not self.is_same_head and 'projection_head.' in newkey:
          newkey1 = newkey.replace('projection_head.', 'projection_head_LPUN.')
          loadstate[newkey1] = state[key]

    self.load_state_dict(loadstate, strict=False)
    return self

  def distribute_model1(self):
      self.feature = nn.DataParallel(self.feature)
      return self

  def forward_alternate_head(self, x):
    r = self.feature(x)
    z = self.projection_head_LPUN(r)
    return z, r

  def LPUN_contrastive_loss(self, x, shots_per_way, n_way, unlab_bsz, mode='standard'):
    # labelled positives and Unlabelled negatives
    bsz,n_s = x.size()[:2]
    n_l = n_way * shots_per_way
    n_ul = unlab_bsz * n_s
    x = x.cuda()
    x = x.contiguous().view(bsz * n_s, *x.size()[2:])
    if self.is_same_head:
      z, _ = self.forward_this(x)
    else:
      z, _ = self.forward_alternate_head(x)

    # positive mask
    T1 = np.eye(int(n_l/n_s))
    T2 = np.ones((n_s, n_s))
    mask_pos_lab = torch.FloatTensor(np.kron(T1, T2))
    T3 = torch.cat([mask_pos_lab, torch.zeros(n_l, n_ul)], dim=1)
    T4 = torch.cat([torch.zeros(n_ul, n_l), torch.zeros(n_ul,n_ul)], dim=1)
    mask_pos = torch.cat([T3,T4], dim=0).to(z.device)

    # negative mask
    mask_neg_lab = torch.zeros(n_l, n_l)
    T3 = torch.cat([mask_neg_lab, torch.ones(n_l, n_ul)], dim=1)
    T4 = torch.ones(n_ul, n_l+n_ul) #dummy
    mask_neg = torch.cat([T3,T4], dim=0).to(z.device)

    if mode=='standard':
      return self.general_contrastive_loss(z, mask_pos, mask_neg, n_s)
    elif mode=='hpm_v1':
      return self.hpm_contrastive_loss_v1(z, mask_pos, mask_neg, n_s)