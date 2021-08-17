# nearest positve contrastive learning
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from methods.weight_imprint_based import WeightImprint
from methods import backbone
from tqdm import tqdm
from tensorboardX import SummaryWriter
import ipdb

EPS=0.00001
class ContrastiveModel(WeightImprint):
  maml = False

  def __init__(self, model_func, tau, mode='vanila', projhead=False, alpha=0.5, beta=1.0,
               tf_path=None, loadpath=None, is_distribute=False):
    super(ContrastiveModel, self).__init__(model_func=model_func, tf_path=tf_path)
    self.method = 'ContrastiveModel'
    self.tau = tau
    self.mode = mode

    if mode=='vanila':
      self.contrastive_loss_fn = self.general_contrastive_loss
    else:
      # hard negative mining or # task alignment based mining
      self.contrastive_loss_fn = self.weighted_contrastive_loss
      self.beta = beta
      if mode=='joint':
        # align + hnm
        self.alpha=alpha

    self.projhead = projhead
    if projhead:
      self.projected_feature_dim = 128
      self.projection_head = nn.Sequential(
        nn.Linear(self.feature.final_feat_dim, self.projected_feature_dim, bias=True)
      )
      print('--- with projection ---')
    else:
      self.projection_head = None
      print('--- No projection ---')

    if loadpath != None:
      self.load_model(loadpath)
    if is_distribute:
      self.distribute_model()

  def refresh_from_chkpt(self):
    self.load_state_dict(self.init_checkpoint, strict=False)
    if self.projhead:
      self.projection_head = nn.Sequential(
        nn.Linear(self.feature.final_feat_dim, self.projected_feature_dim, bias=True)
      )
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

  def get_ft_layers(self, num_layers=0):
    if num_layers==0:
      ret_params = [{'params':self.feature.parameters()}]
    else:
      assert self.model_func==backbone.ResNet10
      for param in self.feature.parameters():
        param.requires_grad = False
      if num_layers==1:
        ret_params =[{'params':self.feature.last_block.BNshortcut.parameters()},
                     {'params':self.feature.last_block.shortcut.parameters()}]
      elif num_layers==2:
        ret_params =[{'params':self.feature.last_block.BNshortcut.parameters()},
                     {'params':self.feature.last_block.shortcut.parameters()},
                     {'params':self.feature.last_block.BN2.parameters()},
                     {'params':self.feature.last_block.C2.parameters()}]
      elif num_layers==3:
        ret_params =[{'params':self.feature.last_block.BNshortcut.parameters()},
                     {'params':self.feature.last_block.shortcut.parameters()},
                     {'params':self.feature.last_block.BN2.parameters()},
                     {'params':self.feature.last_block.C2.parameters()},
                     {'params':self.feature.last_block.BN1.parameters()},
                     {'params':self.feature.last_block.C1.parameters()}]
    if self.projhead:
      ret_params.append({'params':self.projection_head.parameters()})

    for rp in ret_params:
      for param in rp['params']:
        param.requires_grad = True
    return ret_params

  def forward_this(self, x):
    z = self.get_feature(x)
    if self.projhead:
      return self.projection_head(z)
    else:
      return z

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

  def weighted_contrastive_loss(self, z_inp, mask_pos, mask_neg, n_s):
    # normalize
    z = F.normalize(z_inp, dim=1)
    bsz, featdim = z.size()
    z_square = z.view(bsz, 1, featdim).repeat(1, bsz, 1)
    sim = nn.CosineSimilarity(dim=2)(z_square, z_square.transpose(1, 0))
    Sv = torch.exp(sim / self.tau)

    # weighted negative
    if self.mode=='hnm':
      # hard neg mining
      weights = mask_neg * torch.exp(sim / self.beta)
    elif self.mode=='align':
      # task alignment based mining
      task_rep = z_inp[mask_pos.sum(dim=1) > 0,:].mean(dim=0)
      task_rep = F.normalize(task_rep, dim=0)
      task_rep = task_rep.unsqueeze(0).unsqueeze(1).repeat(bsz, bsz, 1)
      alignment = nn.CosineSimilarity(dim=2)(z_square.transpose(1, 0), task_rep)
      weights = mask_neg * torch.exp(alignment / self.beta)
    elif self.mode=='align_neg':
      # select negatives that are aligned with the labelled negatives
      M1 = (mask_pos.sum(dim=1) > 0).unsqueeze(0).unsqueeze(2).float()
      M2 = (mask_pos.sum(dim=1) > 0).unsqueeze(0).unsqueeze(1).float()
      M = M1.bmm(M2).squeeze(0)
      mask_negrep = M * mask_neg
      mask_negrep = mask_negrep.unsqueeze(2)
      z_inp = z_inp.unsqueeze(0).repeat(bsz, 1, 1)
      task_rep = (z_inp * mask_negrep).sum(dim=1)/(mask_negrep.sum(dim=1) + EPS)
      task_rep = F.normalize(task_rep, dim=1)
      task_rep = task_rep.unsqueeze(1).repeat(1, bsz, 1)
      alignment = nn.CosineSimilarity(dim=2)(z_square.transpose(1, 0), task_rep)
      weights = mask_neg * torch.exp(alignment / self.beta)
    elif self.mode=='joint':
      # convex_comb(alignment, hard neg mining)
      task_rep = z_inp[mask_pos.sum(dim=1) > 0, :].mean(dim=0)
      task_rep = task_rep.unsqueeze(0).repeat(bsz,1)
      joint_rep = self.alpha * z_inp + (1-self.alpha) * task_rep
      joint_rep = F.normalize(joint_rep, dim=1)
      # print('sim of each datapt wrt taskrep',nn.CosineSimilarity(dim=1)(z, task_rep)[:20])
      # print('sim of jointrep wrt taskrep',nn.CosineSimilarity(dim=1)(joint_rep, task_rep)[:20])
      # print('sim of jointrep wrt datapt',nn.CosineSimilarity(dim=1)(joint_rep, z)[:20])
      joint_rep = joint_rep.unsqueeze(1).repeat(1, bsz, 1)
      alignment = nn.CosineSimilarity(dim=2)(z_square.transpose(1, 0), joint_rep)
      weights = mask_neg * torch.exp(alignment / self.beta)
    weight_factor = weights / weights.sum(dim=1).unsqueeze(1).repeat(1, bsz)
    weight_factor = weight_factor.detach()
    # negative term is actually a sum and not mean, hence the unnormalization
    neg = ((Sv*weight_factor).sum(dim=1) * mask_neg.sum(dim=1)).unsqueeze(1).repeat(1, bsz)
    li = mask_pos * torch.log(Sv / (Sv + neg) + EPS)
    li = li - li.diag().diag()
    li = (1 / (n_s - 1)) * li.sum(dim=1)
    loss = -li[mask_pos.sum(dim=1) > 0].mean()
    return loss

  def OL_contrastive_loss(self, x, shots_per_way, n_way):
    bsz,n_s = x.size()[:2]
    x = x.contiguous().view(bsz * n_s, *x.size()[2:])
    z = self.forward_this(x)
    # positive mask
    T1 = np.eye(bsz)
    T2 = np.ones((n_s, n_s))
    mask_pos = torch.FloatTensor(np.kron(T1, T2)).to(z.device)
    # negative mask
    T1 = 1-np.eye(n_way)
    T2 = np.ones((shots_per_way, shots_per_way))
    mask_neg = torch.FloatTensor(np.kron(T1, T2)).to(z.device)
    return self.general_contrastive_loss(z, mask_pos, mask_neg, n_s)

  def OL_hpm_loss(self, z, shots_per_way, n_way):
    n_s = 2
    bsz = z.size(0)
    # positive mask
    T1 = np.eye(int(bsz/n_s))
    T2 = np.zeros((n_s, n_s))
    T2[0,1] = 1
    mask_pos = torch.FloatTensor(np.kron(T1, T2)).to(z.device)
    # negative mask
    T1 = 1-np.eye(n_way)
    T2 = np.ones((shots_per_way, shots_per_way))
    mask_neg = torch.FloatTensor(np.kron(T1, T2)).to(z.device)
    return self.general_contrastive_loss(z, mask_pos, mask_neg, n_s)

  def LPAN_contrastive_loss(self, z, shots_per_way, n_way, n_ul):
    # labelled positives and all negatives
    n_pos = 2
    n_l = n_way * shots_per_way
    # z = self.forward_this(x)
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
    return self.contrastive_loss_fn(z, mask_pos, mask_neg, n_pos)


  def LPAN_hpm_loss(self, z, shots_per_way, n_way, n_ul):
    # labelled positives and all negatives
    n_pos = 2
    n_l = n_way * shots_per_way
    # positive mask
    T1 = np.eye(int(n_l/n_pos))
    T2 = np.zeros((n_pos, n_pos))
    T2[0,1] = 1
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
    return self.contrastive_loss_fn(z, mask_pos, mask_neg, n_pos)



  # def LPAN_contrastive_loss(self, x, shots_per_way, n_way, unlab_bsz):
  #   # labelled positives and all negatives
  #   bsz,n_s = x.size()[:2]
  #   n_l = n_way * shots_per_way
  #   n_ul = unlab_bsz * n_s
  #   x = x.cuda()
  #   x = x.contiguous().view(bsz * n_s, *x.size()[2:])
  #   z = self.forward_this(x)
  #
  #   # positive mask
  #   T1 = np.eye(int(n_l/n_s))
  #   T2 = np.ones((n_s, n_s))
  #   mask_pos_lab = torch.FloatTensor(np.kron(T1, T2))
  #   T3 = torch.cat([mask_pos_lab, torch.zeros(n_l, n_ul)], dim=1)
  #   T4 = torch.cat([torch.zeros(n_ul, n_l), torch.zeros(n_ul,n_ul)], dim=1)
  #   mask_pos = torch.cat([T3,T4], dim=0).to(z.device)
  #
  #   # negative mask
  #   T1 = 1-np.eye(n_way)
  #   T2 = np.ones((shots_per_way, shots_per_way))
  #   mask_neg_lab = torch.FloatTensor(np.kron(T1, T2))
  #   T3 = torch.cat([mask_neg_lab, torch.ones(n_l, n_ul)], dim=1)
  #   T4 = torch.ones(n_ul, n_l + n_ul)  # dummy
  #   mask_neg = torch.cat([T3,T4], dim=0).to(z.device)
  #
  #   return self.contrastive_loss_fn(z, mask_pos, mask_neg, n_s)
