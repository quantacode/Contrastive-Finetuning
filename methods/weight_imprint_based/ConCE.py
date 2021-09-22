import numpy as np
from tqdm import tqdm
import ipdb

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

from methods.weight_imprint_based import WeightImprint
from methods import backbone
from tensorboardX import SummaryWriter

from utils import *

EPS=0.00001
def weight_reset(m):
  if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
    m.reset_parameters()

class ConCeModel(WeightImprint):
  maml = False

  def __init__(self, model_func, tau, n_way, tf_path=None, loadpath=None, projhead=False,
               is_distribute=False, src_classes=0, ft_mode=False, cos_fac=1.0):
    self.tau = tau
    self.n_way = n_way
    self.src_classes = src_classes
    self.cos_fac = cos_fac
    self.model_func = model_func
    self.tf_path = tf_path
    self.projhead = projhead

    # primary head
    self.ft_mode = ft_mode
    self.loadpath = loadpath
    self.is_distribute = is_distribute
    self.init_parameters()
    self.init_model()

  def init_parameters(self):
    self.init_checkpoint=None
    if self.projhead:
      self.projected_feature_dim = 64
      print('--- with projection %d ---'%(self.projected_feature_dim))
    else:
      print('--- No projection ---')

    if 'mtce' in self.ft_mode:
      print('--- Secondary classifier indim ---')

  def init_model(self):
    super(ConCeModel, self).__init__(model_func=self.model_func, tf_path=self.tf_path)
    if self.ft_mode == 'ce_mtce' or self.ft_mode == 'ce':
      self.L = weight_norm(nn.Linear(self.feat_dim, self.n_way, bias=False), name='weight', dim=0)
      self.projection_head = None
    else:
      if self.projhead:
        self.projection_head = nn.Sequential(
          nn.Linear(self.feat_dim, self.projected_feature_dim, bias=True)
        )
      else:
        self.projection_head = None

    # secondary head
    if 'mtce' in self.ft_mode:
      self.source_L = weight_norm(nn.Linear(self.feat_dim, self.src_classes, bias=False), name='weight', dim=0)

    self.loss_fn = nn.CrossEntropyLoss()
    if self.loadpath != None:
      self.load_model()
    if self.is_distribute:
      self.distribute_model()

  def load_model(self):
    if self.init_checkpoint is not None:
      self.load_state_dict(self.init_checkpoint, strict=False)
    else:
      state = torch.load(self.loadpath)
      loadstate = {}
      if 'state' in state.keys():
        state = state['state']
        for key in state.keys():
          if 'feature.module' in key:
            loadstate[key.replace('feature.module', 'feature')] = state[key]
          else:
            loadstate[key] = state[key]
      elif 'state_dict' in state.keys():
        state = state['state_dict']
        for key in state.keys():
          if 'module.encoder_k' in key:
            loadstate[key.replace('module.encoder_k', 'feature')] = state[key]
          else:
            loadstate[key] = state[key]
      self.init_checkpoint = loadstate
      self.load_state_dict(loadstate, strict=False)
    return self


  def refresh_from_chkpt(self):
    self.init_model()

  def distribute_model(self):
    self.feature = nn.DataParallel(self.feature)
    return self

  def forward_projection(self, z):
    if self.projection_head is not None:
      return self.projection_head(z)
    else:
      return z

  def forward_this(self, x):
    return self.forward_projection(self.get_feature(x))

  def ewn_contrastive_loss(self, z, mask_pos, mask_neg, mask_distract, n_s, alpha):
    # equally weighted task and distractor negative contrastive loss
    bsz, featdim = z.size()
    z_square = z.view(bsz, 1, featdim).repeat(1, bsz, 1)
    sim = nn.CosineSimilarity(dim=2)(z_square, z_square.transpose(1, 0))
    Sv = torch.exp(sim / self.tau)
    neg = (Sv * mask_neg)
    neg = alpha*(1-mask_distract)*neg + (1-alpha)*mask_distract*neg
    neg = 2*neg
    neg = neg.sum(dim=1).unsqueeze(1).repeat(1, bsz)
    li = mask_pos * torch.log(Sv / (Sv + neg) + EPS)
    li = li - li.diag().diag()
    li = (1 / (n_s - 1)) * li.sum(dim=1)
    loss = -li[mask_pos.sum(dim=1) > 0].mean()
    return loss

  #################  MAIN ##############
  def cssf_loss(self, z, shots_per_way, n_way, n_ul, mode='lpan', alpha=None):
    # labelled positives and all negatives
    n_pos = 2
    n_l = n_way * shots_per_way
    # positive mask
    T1 = np.eye(int(n_l/n_pos))
    T2 = np.ones((n_pos, n_pos))
    mask_pos_lab = torch.FloatTensor(np.kron(T1, T2))
    T3 = torch.cat([mask_pos_lab, torch.zeros(n_l, n_ul)], dim=1)
    T4 = torch.zeros(n_ul, n_l+n_ul)
    mask_pos = torch.cat([T3,T4], dim=0).to(z.device)
    # negative mask
    T1 = 1-np.eye(n_way)
    T2 = np.ones((shots_per_way, shots_per_way))
    mask_neg_lab = torch.FloatTensor(np.kron(T1, T2))
    T3 = torch.cat([mask_neg_lab, torch.ones(n_l, n_ul)], dim=1)
    T4 = torch.ones(n_ul, n_l + n_ul)  # dummy
    mask_neg = torch.cat([T3,T4], dim=0).to(z.device)
    T3 = torch.cat([torch.zeros(n_l, n_l), torch.ones(n_l, n_ul)], dim=1)
    mask_distract = torch.cat([T3, T4], dim=0).to(z.device)
    alpha = n_ul/(n_ul + n_l - shots_per_way)
    return self.ewn_contrastive_loss(z, mask_pos, mask_neg, mask_distract, n_pos, alpha)

  #################  MAIN ##############

  def get_classification_scores(self, z, classifier):
    z_norm = torch.norm(z, p=2, dim=1).unsqueeze(1).expand_as(z)
    z_normalized = z.div(z_norm + EPS)
    L_norm = torch.norm(classifier.weight.data, p=2, dim=1).unsqueeze(1).expand_as(classifier.weight.data)
    classifier.weight.data = classifier.weight.data.div(L_norm + EPS)
    cos_dist = classifier(z_normalized)
    scores = self.cos_fac * cos_dist
    return scores

  def CE_loss(self, x, y):
    z = self.get_feature(x)
    scores = self.get_classification_scores(z, self.L)
    loss = self.loss_fn(scores, y)
    return loss

  def CE_loss_source(self, z, y):
    scores = self.get_classification_scores(z, self.source_L)
    loss = self.loss_fn(scores, y)
    return loss

  def get_linear_classification_scores(self, z, classifier):
    scores = classifier(z)
    return scores

  def LCE_loss(self, x, y):
    z = self.get_feature(x)
    scores = self.get_linear_classification_scores(z, self.L)
    loss = self.loss_fn(scores, y)
    return loss
