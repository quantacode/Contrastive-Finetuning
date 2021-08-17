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
from utils import *
import ipdb

EPS=0.00001
def weight_reset(m):
  if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
    m.reset_parameters()

class ConCeModel(WeightImprint):
  maml = False

  def __init__(self, model_func, tau, n_way, tf_path=None, loadpath=None, projhead=False,
               is_distribute=False, src_classes=0, ft_mode=False, cos_fac=1.0):
    # super(ConCeModel, self).__init__(model_func=model_func, tf_path=tf_path)
    self.method = 'ConCeModel'
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
    if self.ft_mode == 'ce_mtce' or self.ft_mode == 'ce':
      print('--- Primary Classifier --')
    else:
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
      if 'model' in state:
        # resnet12 model load from rfs paper
        state = state['model']
        for key in self.state_dict().keys():
          if key.replace('feature.', '') in state.keys():
            loadstate[key] = state[key.replace('feature.', '')]
      elif 'state' in state.keys():
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
      # return self.refresh_from_chkpt()

      # chkpt_vis(self, -10, -1)
      self.load_state_dict(loadstate, strict=False)
      # chkpt_vis(self, -10, -1)
      # ipdb.set_trace()
    return self


  def refresh_from_chkpt(self):
    self.init_model()

  # def load_model(self, loadpath):
  #   state = torch.load(loadpath)
  #   if 'state' in state.keys():
  #     state = state['state']
  #   loadstate = {}
  #   for key in state.keys():
  #     if 'feature.module' in key:
  #       loadstate[key.replace('feature.module', 'feature')] = state[key]
  #     else:
  #       loadstate[key] = state[key]
  #   self.init_checkpoint = loadstate
  #   return self.refresh_from_chkpt()

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
    z = F.normalize(z, dim=1)
    bsz, featdim = z.size()
    z_square = z.view(bsz, 1, featdim).repeat(1, bsz, 1)
    Sv = nn.CosineSimilarity(dim=2)(z_square, z_square.transpose(1, 0))
    Sv = torch.exp(Sv / self.tau)
    neg = (Sv * mask_neg).sum(dim=1).unsqueeze(1).repeat(1, bsz)
    li = mask_pos * torch.log(Sv / (Sv + neg) + EPS)
    li = li - li.diag().diag()
    li = (1 / (n_s - 1)) * li.sum(dim=1)
    loss = -li[mask_pos.sum(dim=1) > 0].mean()
    return loss

  def rw_contrastive_loss(self, z, mask_pos, mask_neg, mask_distract, n_s, alpha):
    # reweighted contrastive loss
    # normalize
    z = F.normalize(z, dim=1)
    bsz, featdim = z.size()
    z_square = z.view(bsz, 1, featdim).repeat(1, bsz, 1)
    sim = nn.CosineSimilarity(dim=2)(z_square, z_square.transpose(1, 0))
    Sv = torch.exp(sim / self.tau)
    neg = (Sv * mask_neg)
    weight = torch.exp(-sim/alpha) * mask_distract
    weight = weight / weight.sum(dim=1).unsqueeze(1)
    # relative negative weight multiplier
    rnw_mult = mask_distract.sum(dim=1).unsqueeze(1)
    neg = (1-mask_distract)*neg + rnw_mult*weight.detach()*neg
    neg = neg.sum(dim=1).unsqueeze(1).repeat(1, bsz)
    li = mask_pos * torch.log(Sv / (Sv + neg) + EPS)
    li = li - li.diag().diag()
    li = (1 / (n_s - 1)) * li.sum(dim=1)
    loss = -li[mask_pos.sum(dim=1) > 0].mean()
    return loss

  def tnw_contrastive_loss(self, z, mask_pos, mask_neg, proto_task_neg, n_s, alpha):
    # reweighted contrastive loss
    # normalize
    z = F.normalize(z, dim=1)
    bsz, featdim = z.size()
    z_square = z.view(bsz, 1, featdim).repeat(1, bsz, 1)
    sim = nn.CosineSimilarity(dim=2)(z_square, z_square.transpose(1, 0))
    Sv = torch.exp(sim / self.tau)
    neg = (Sv * mask_neg)

    # reshape taskneg according to the z_square
    proto_task_neg = proto_task_neg.unsqueeze(1).repeat(1, bsz, 1)
    taskneg_sim = nn.CosineSimilarity(dim=2)(z_square)
    weight = torch.exp(-sim/alpha) * mask_distract
    weight = weight / weight.sum(dim=1).unsqueeze(1)
    # relative negative weight multiplier
    rnw_mult = mask_distract.sum(dim=1).unsqueeze(1)
    neg = (1-mask_distract)*neg + rnw_mult*weight.detach()*neg
    neg = neg.sum(dim=1).unsqueeze(1).repeat(1, bsz)
    li = mask_pos * torch.log(Sv / (Sv + neg) + EPS)
    li = li - li.diag().diag()
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

  def cso_loss(self, z, shots_per_way, n_way):
    # contrastive supervised only loss
    n_pos = 2
    n_l = n_way * shots_per_way
    # positive mask
    T1 = np.eye(int(n_l / n_pos))
    T2 = np.ones((n_pos, n_pos))
    mask_pos = torch.FloatTensor(np.kron(T1, T2)).to(z.device)
    # negative mask
    T1 = 1-np.eye(n_way)
    T2 = np.ones((shots_per_way, shots_per_way))
    mask_neg = torch.FloatTensor(np.kron(T1, T2)).to(z.device)
    return self.general_contrastive_loss(z, mask_pos, mask_neg, n_pos)

  def subd_cssf_loss(self, z_l, z_u, shots_per_way, n_way, n_ul, mode):
    # subsampled distractors
    n_subd = 64
    n_pos = 2
    n_l = n_way * shots_per_way
    # positive mask
    T1 = np.eye(int(n_l / n_pos))
    T2 = np.ones((n_pos, n_pos)) - np.eye(n_pos)
    mask_pos_lab = torch.FloatTensor(np.kron(T1, T2))
    mask_pos = torch.cat([mask_pos_lab, torch.zeros(n_l, n_subd)], dim=1).to(z_l.device)
    # negative mask
    T1 = 1 - np.eye(n_way)
    T2 = np.ones((shots_per_way, shots_per_way))
    mask_neg_lab = torch.FloatTensor(np.kron(T1, T2))
    mask_neg = torch.cat([mask_neg_lab, torch.ones(n_l, n_subd)], dim=1).to(z_l.device)
    # 1) closest to task
    # 2) closest to task neg
    # 3) closest to task pos
    if '1' in mode:
      proto_task = z_l.mean(dim=0).unsqueeze(0).repeat(n_ul,1)
      sim = nn.CosineSimilarity(dim=1)(proto_task, z_u)
      z_u_sub = z_u[torch.argsort(sim)[-n_subd:]]
      z_u_sub = z_u_sub.unsqueeze(0).repeat(n_l, 1, 1)
    elif '2' in mode:
      sum_all_z = z_l.sum(dim=0).view(1,-1).repeat(n_way,1)
      sum_pos_z = z_l.view(n_way, shots_per_way, z_l.shape[-1]).sum(dim=1)
      proto_taskneg = (sum_all_z - sum_pos_z)/(n_l-shots_per_way)
      proto_taskneg = proto_taskneg.unsqueeze(1).repeat(1, shots_per_way, 1).view(-1, proto_taskneg.shape[-1])
      t1 = proto_taskneg.unsqueeze(1).repeat(1, n_ul, 1)
      t2 = z_u.unsqueeze(0).repeat(n_l, 1, 1)
      sim = nn.CosineSimilarity(dim=2)(t1, t2)
      subd_idx = torch.argsort(sim, dim=1)[:,-n_subd:].unsqueeze(2).repeat(1,1,t2.shape[2])
      z_u_sub = torch.gather(t2, dim=1, index=subd_idx)
    elif '3' in mode:
      proto_taskpos = z_l.view(n_way, shots_per_way, z_l.shape[-1]).mean(dim=1)
      proto_taskpos = proto_taskpos.unsqueeze(1).repeat(1, shots_per_way, 1).view(-1, proto_taskpos.shape[-1])
      t1 = proto_taskpos.unsqueeze(1).repeat(1, n_ul, 1)
      t2 = z_u.unsqueeze(0).repeat(n_l, 1, 1)
      sim = nn.CosineSimilarity(dim=2)(t1, t2)
      subd_idx = torch.argsort(sim, dim=1)[:,-n_subd:].unsqueeze(2).repeat(1,1,t2.shape[2])
      z_u_sub = torch.gather(t2, dim=1, index=subd_idx)
    else:
      NotImplementedError
    bsz = n_l+n_subd
    z_l_x = z_l.unsqueeze(1).repeat(1, n_l, 1)
    z_l_y = z_l_x.transpose(1, 0)
    z_y = torch.cat([z_l_y, z_u_sub], dim=1)
    z_x = z_l.unsqueeze(1).repeat(1, bsz, 1)
    Sv = nn.CosineSimilarity(dim=2)(z_x, z_y)
    Sv = torch.exp(Sv / self.tau)
    neg = (Sv * mask_neg).sum(dim=1).unsqueeze(1).repeat(1, bsz)
    li = mask_pos * torch.log(Sv / (Sv + neg) + EPS)
    li = (1 / (n_pos - 1)) * li.sum(dim=1)
    loss = -li.mean()
    return loss

  def cssfCB_loss(self, z, n_way, n_support, n_ul, mode):
    # labelled positives and all negatives
    n_l = n_way * n_support
    # positive mask
    T1 = np.eye(n_way)
    T2 = np.ones((n_support, n_support))
    mask_pos_lab = torch.FloatTensor(np.kron(T1, T2))
    T3 = torch.cat([mask_pos_lab, torch.zeros(n_l, n_ul)], dim=1)
    T4 = torch.zeros(n_ul, n_l+n_ul)
    mask_pos = torch.cat([T3,T4], dim=0).to(z.device)
    # negative mask
    if 'lpan' in mode:
      mask_neg = 1-mask_pos
    elif 'lpun' in mode:
      mask_neg_lab = torch.zeros(n_l, n_l)
      T3 = torch.cat([mask_neg_lab, torch.ones(n_l, n_ul)], dim=1)
      T4 = torch.ones(n_ul, n_l + n_ul)  # dummy
      mask_neg = torch.cat([T3, T4], dim=0).to(z.device)

    if 'ewn_lpan' in mode:
      T3 = torch.cat([torch.zeros(n_l, n_l), torch.ones(n_l, n_ul)], dim=1)
      mask_distract = torch.cat([T3, torch.ones(n_ul, n_l+n_ul)], dim=0).to(z.device)
      alpha = n_ul/(n_ul + n_l - n_support)
      return self.ewn_contrastive_loss(z, mask_pos, mask_neg, mask_distract, n_support, alpha)
    else:
      return self.general_contrastive_loss(z, mask_pos, mask_neg, n_support)

  def APAN_cssfCB_loss(self, z, y, n_way, n_support, n_ul):
    # labelled positives and all negatives
    n_l = n_way * n_support
    # positive masks
    mask_pos = (y.unsqueeze(1).repeat(1,n_l+n_ul) == y.unsqueeze(0).repeat(n_l+n_ul,1)).float()
    # negative mask
    mask_neg = 1-mask_pos
    #loss
    bsz, featdim = z.size()
    z_square = z.view(bsz, 1, featdim).repeat(1, bsz, 1)
    Sv = nn.CosineSimilarity(dim=2)(z_square, z_square.transpose(1, 0))
    Sv = torch.exp(Sv / self.tau)
    neg = (Sv * mask_neg).sum(dim=1).unsqueeze(1).repeat(1, bsz)
    li = mask_pos * torch.log(Sv / (Sv + neg) + EPS)
    li = li - li.diag().diag()
    # -1 accounts for the exclusion of self-similarity
    num_pos = mask_pos.sum(dim=1) - 1
    li = li.sum(dim=1)
    li = -li[num_pos>0]/num_pos[num_pos>0]
    loss = li.mean()
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
    if 'lpan' in mode:
      T1 = 1-np.eye(n_way)
      T2 = np.ones((shots_per_way, shots_per_way))
      mask_neg_lab = torch.FloatTensor(np.kron(T1, T2))
    elif 'lpun' in mode:
      mask_neg_lab = torch.zeros(n_l, n_l)
    T3 = torch.cat([mask_neg_lab, torch.ones(n_l, n_ul)], dim=1)
    T4 = torch.ones(n_ul, n_l + n_ul)  # dummy
    mask_neg = torch.cat([T3,T4], dim=0).to(z.device)
    # if mode=='rw_lpan':
    #   T3 = torch.
    #   cat([torch.zeros(n_l, n_l), torch.ones(n_l, n_ul)], dim=1)
    #   mask_distract = torch.cat([T3, T4], dim=0).to(z.device)
    #   return self.rw_contrastive_loss(z, mask_pos, mask_neg, mask_distract, n_pos, alpha)
    # elif mode=='tnw_lpan':
    #   ipdb.set_trace()
    #   # first check subsampling distractors according to
    #   # 1) closest to task
    #   # 2) closest to task neg
    #   # 3) closest to task pos
    #   # Use the best criteria for soft reweighting here
    #   summ_task_z = z[:n_l].sum(dim=0).unsqueeze(0).repeat(n_way,1)
    #   # per class aggregate
    #   z_pc = z.contiguous().view(n_way, shots_per_way, z.shape[-1]).sum(dim=1)
    #   # criteria for reweighting
    #   proto_taskneg = (summ_task_z - z_pc)/(n_l-shots_per_way)
    #   proto_taskneg = proto_taskneg.unsqueeze(1).repeat(1,shots_per_way,1).view(-1, proto_taskneg.shape[-1]).sum(dim=1)
    #   return self.tnw_contrastive_loss(z, mask_pos, mask_neg, proto_taskneg, n_pos, alpha)
    # elif mode=='ewn_lpan':
    if 'ewn' in mode:
      T3 = torch.cat([torch.zeros(n_l, n_l), torch.ones(n_l, n_ul)], dim=1)
      mask_distract = torch.cat([T3, T4], dim=0).to(z.device)
      alpha = n_ul/(n_ul + n_l - shots_per_way)
      return self.ewn_contrastive_loss(z, mask_pos, mask_neg, mask_distract, n_pos, alpha)
    else:
      return self.general_contrastive_loss(z, mask_pos, mask_neg, n_pos)


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



#########################################################

  # def tar_cssf_loss(self, z_l, z_u, shots_per_way, n_way, n_ul, alpha=None):
  #   n_pos = 2
  #   n_l = n_way * shots_per_way
  #   # positive mask
  #   T1 = np.eye(int(n_l / n_pos))
  #   T2 = np.ones((n_pos, n_pos))
  #   mask_pos_lab = torch.FloatTensor(np.kron(T1, T2))
  #   T3 = torch.cat([mask_pos_lab, torch.zeros(n_l, n_ul)], dim=1)
  #   T4 = torch.zeros(n_ul, n_l + n_ul)
  #   mask_pos = torch.cat([T3, T4], dim=0).to(z_l)
  #   # negative mask
  #   T1 = 1 - np.eye(n_way)
  #   T2 = np.ones((shots_per_way, shots_per_way))
  #   mask_neg_lab = torch.FloatTensor(np.kron(T1, T2))
  #   T3 = torch.cat([mask_neg_lab, torch.ones(n_l, n_ul)], dim=1)
  #   T4 = torch.ones(n_ul, n_l + n_ul)  # dummy
  #   mask_neg = torch.cat([T3, T4], dim=0).to(z_l)
  #
  #   # contrastive loss
  #   proto_task = z_l.mean(dim=0)
  #   z = torch.cat([z_l, z_u], dim=0)
  #   bsz, featdim = z.size()
  #   z_square = z.view(bsz, 1, featdim).repeat(1, bsz, 1)
  #   sim = nn.CosineSimilarity(dim=2)(z_square, z_square.transpose(1, 0))
  #   Sv = torch.exp(sim / self.tau)
  #   neg = (Sv * mask_neg)
  #   # reshape task proto according to the z_square
  #   proto_task = proto_task.view(1,1,-1).repeat(bsz, bsz, 1)
  #   task_alignment = nn.CosineSimilarity(dim=2)(z_square.transpose(1, 0), proto_task)
  #   weight = torch.exp(task_alignment/alpha) * mask_neg
  #   weight = weight / weight.sum(dim=1).unsqueeze(1)
  #   neg = weight.detach() * neg
  #   neg = neg.sum(dim=1).unsqueeze(1).repeat(1, bsz)
  #   li = mask_pos * torch.log(Sv / (Sv + (n_l + n_ul - shots_per_way) * neg) + EPS)
  #   li = li - li.diag().diag()
  #   li = (1 / (n_pos - 1)) * li.sum(dim=1)
  #   loss = -li[mask_pos.sum(dim=1) > 0].mean()
  #   return loss

  #
  # def supcon_loss(self, z, n_way, n_support, n_ul):
  #   # labelled positives and all negatives
  #   n_l = n_way * n_support
  #   # positive mask
  #   T1 = np.eye(n_way)
  #   T2 = np.ones((n_support, n_support))
  #   mask_pos_lab = torch.FloatTensor(np.kron(T1, T2))
  #   T3 = torch.cat([mask_pos_lab, torch.zeros(n_l, n_ul)], dim=1)
  #   T4 = torch.zeros(n_ul, n_l+n_ul)
  #   mask_pos = torch.cat([T3,T4], dim=0).to(z.device)
  #   # supcon implementation
  #   z = F.normalize(z, dim=1)
  #   bsz, featdim = z.size()
  #   z_square = z.view(bsz, 1, featdim).repeat(1, bsz, 1)
  #   Sv = nn.CosineSimilarity(dim=2)(z_square, z_square.transpose(1, 0))
  #   Sv = torch.exp(Sv / self.tau)
  #   den = (Sv-Sv.diag().diag()).sum(dim=1).unsqueeze(1).repeat(1, bsz)
  #   li = mask_pos * torch.log(Sv / den + EPS)
  #   li = li - li.diag().diag()
  #   li = (1 / (n_support - 1)) * li.sum(dim=1)
  #   loss = -li[mask_pos.sum(dim=1) > 0].mean()
  #   return loss