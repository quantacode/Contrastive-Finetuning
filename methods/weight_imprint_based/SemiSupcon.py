# nearest positve contrastive learning
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
import numpy as np
from methods.weight_imprint_based import WeightImprint
from methods import backbone
from tqdm import tqdm
from tensorboardX import SummaryWriter
import ipdb

EPS=0.00001
class SSconModel(WeightImprint):
  maml = False

  def __init__(self, model_func, tau, n_way, tf_path=None, loadpath=None, projhead=False,
               is_distribute=False, src_classes=0, ft_mode=False):
    super(SSconModel, self).__init__(model_func=model_func, tf_path=tf_path)
    self.method = 'SSconModel'
    self.tau = tau
    self.n_way = n_way
    self.src_classes = src_classes

    # primary head
    self.ft_mode = ft_mode
    if self.ft_mode == 'ce_mtce' or self.ft_mode == 'ce':
      self.L = weight_norm(nn.Linear(self.feature.final_feat_dim, self.n_way, bias=False), name='weight', dim=0)
      self.projection_head = None
      print('--- Classifier indim : %d ---'%(self.feature.final_feat_dim))
    else:
      if projhead:
        self.projected_feature_dim = 64
        self.projection_head = nn.Sequential(
          nn.Linear(self.feature.final_feat_dim, self.projected_feature_dim, bias=True)
        )
        print('--- with projection %d ---'%(self.projected_feature_dim))
      else:
        self.projection_head = None
        print('--- No projection ---')

    # secondary head
    if 'mtce' in self.ft_mode:
      print('--- Secondary classifier indim : %d ---'%(self.feature.final_feat_dim))
      self.source_L = weight_norm(nn.Linear(self.feature.final_feat_dim, self.src_classes, bias=False), name='weight', dim=0)

    self.loss_fn = nn.CrossEntropyLoss()
    if loadpath != None: self.load_model(loadpath)
    if is_distribute: self.distribute_model()

  def refresh_from_chkpt(self):
    self.load_state_dict(self.init_checkpoint, strict=False)
    if self.ft_mode == 'ce_mtce' or self.ft_mode == 'ce':
      self.L = weight_norm(nn.Linear(self.feature.final_feat_dim, self.n_way, bias=False), name='weight', dim=0)
    else:
      if self.projection_head is not None:
        self.projection_head = nn.Sequential(
          nn.Linear(self.feature.final_feat_dim, self.projected_feature_dim, bias=True)
        )
    if 'mtce' in self.ft_mode:
      self.source_L = weight_norm(nn.Linear(self.feature.final_feat_dim, self.src_classes, bias=False), name='weight', dim=0)
    return self

  def load_model(self, loadpath):
    state = torch.load(loadpath)
    if 'state' in state.keys():
      state = state['state']
    loadstate = {}
    for key in state.keys():
      if 'feature.module' in key:
        loadstate[key.replace('feature.module', 'feature')] = state[key]
      else:
        loadstate[key] = state[key]
    self.init_checkpoint = loadstate
    return self.refresh_from_chkpt()

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

  def general_contrastive_loss(self, z, mask_pos, mask_neg, n_s):
    # normalize
    # z = F.normalize(z, dim=1)
    bsz, featdim = z.size()
    z_square = z.view(bsz, 1, featdim).repeat(1, bsz, 1)
    Sv = nn.CosineSimilarity(dim=2)(z_square, z_square.transpose(1, 0))
    Sv = torch.exp(Sv / self.tau)
    neg = (Sv * mask_neg).sum(dim=1).unsqueeze(1).repeat(1, bsz)
    li = mask_pos * torch.log(Sv / neg + EPS)
    li = (1 / (n_s - 1)) * li.sum(dim=1)
    loss = -li[mask_pos.sum(dim=1) > 0].mean()
    return loss

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

  def cssf_loss(self, z, shots_per_way, n_way, n_ul, mode='lpan', alpha=None):
    # labelled positives and all negatives
    n_l = n_way * shots_per_way
    # positive mask
    T1 = np.eye(n_way)
    T2 = np.ones((shots_per_way, shots_per_way))
    mask_pos_lab = torch.FloatTensor(np.kron(T1, T2))
    T3 = torch.cat([mask_pos_lab, torch.zeros(n_l, n_ul)], dim=1)
    T4 = torch.zeros(n_ul, n_l + n_ul)
    mask_pos = torch.cat([T3, T4], dim=0).to(z.device)
    # negative mask
    mask_neg = 1 - mask_pos
    T3 = torch.cat([torch.zeros(n_l, n_l), torch.ones(n_l, n_ul)], dim=1)
    mask_distract = torch.cat([T3, torch.ones(n_ul, n_l + n_ul)], dim=0).to(z.device)
    alpha = n_ul / (n_ul + n_l - shots_per_way)
    return self.ewn_contrastive_loss(z, mask_pos, mask_neg, mask_distract, shots_per_way, alpha)

  def get_classification_scores(self, z, classifier):
    z_norm = torch.norm(z, p=2, dim=1).unsqueeze(1).expand_as(z)
    z_normalized = z.div(z_norm + EPS)
    L_norm = torch.norm(classifier.weight.data, p=2, dim=1).unsqueeze(1).expand_as(classifier.weight.data)
    classifier.weight.data = classifier.weight.data.div(L_norm + EPS)
    cos_dist = classifier(z_normalized)
    scores = 10 * cos_dist
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


  # def cssf_loss(self, z, shots_per_way, n_way, n_ul, mode='lpan', alpha=None):
  #   # includes all samples except for the anchor in denominator (original supcon)
  #   n_l = n_way * shots_per_way
  #   diag_maskout = 1-torch.eye(n_l)
  #   T1 = np.eye(n_way)
  #   T2 = np.ones((shots_per_way, shots_per_way))
  #   mask_pos_lab = diag_maskout * torch.FloatTensor(np.kron(T1, T2))
  #   # positive mask
  #   mask_pos = torch.cat([mask_pos_lab, torch.zeros(n_l, n_ul)], dim=1)
  #   mask_pos = torch.cat([mask_pos, torch.zeros(n_ul, n_l+n_ul)], dim=0).to(z.device)
  #   # negative mask
  #   mask_neg = torch.cat([diag_maskout, torch.ones(n_l, n_ul)], dim=1)
  #   mask_neg = torch.cat([mask_neg, torch.ones(n_ul, n_l + n_ul)], dim=0).to(z.device)
  #
  #   if 'ewn_lpan' in mode:
  #     T3 = torch.cat([torch.zeros(n_l, n_l), torch.ones(n_l, n_ul)], dim=1)
  #     mask_distract = torch.cat([T3, torch.ones(n_ul, n_l + n_ul)], dim=0).to(z.device)
  #     alpha = n_ul/(n_ul + n_l - shots_per_way)
  #     return self.ewn_contrastive_loss(z, mask_pos, mask_neg, mask_distract, shots_per_way, alpha)
  #   else:
  #     return self.general_contrastive_loss(z, mask_pos, mask_neg, shots_per_way)
