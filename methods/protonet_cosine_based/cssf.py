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
from methods.protonet_cosine_based.HPM_utils import get_hp_indices
import random
import ipdb
eps=0.000001
class CSSF(SIMCLR):
  def __init__(self, model_func, tau, n_way, tf_path, loadpath=None, is_distribute=False,
               beta=0, clstau=0, n_trim=1):
    super(CSSF, self).__init__(model_func=model_func,
                               tau=tau, n_way=n_way,
                               tf_path=tf_path, loadpath=loadpath,
                               is_distribute=is_distribute)
    # hard positive mining parameter
    self.beta = beta
    self.clstau = clstau
    self.n_trim = n_trim
    self.method = 'CSSF'

  def general_contrastive_loss(self, z, mask_pos, mask_neg, n_s):
    # normalize
    z = F.normalize(z, dim=1)
    bsz, featdim = z.size()
    z_square = z.view(bsz, 1, featdim).repeat(1, bsz, 1)
    Sv = nn.CosineSimilarity(dim=2)(z_square, z_square.transpose(1, 0))
    Sv = torch.exp(Sv / self.tau)
    neg = (Sv*mask_neg).sum(dim=1).unsqueeze(1).repeat(1,bsz)
    li = mask_pos*torch.log(Sv/(Sv+neg)+eps)
    li = li-li.diag().diag()
    li = (1/(n_s-1))*li.sum(dim=1)
    # Note: dividing li.sum by mask.sum takes care for different strategies of incuding paired-positives in the loss
    # in the vanila case (simclr), the operation boils down to li.mean
    active_anchors = mask_pos.sum(dim=1)>0
    loss = -li.sum(dim=0)/active_anchors.sum(dim=0)
    return loss

  def _get_denominator(self, Sv, neg, sim, mask_neg, label_debug = None):
    weight = torch.exp(-self.beta * sim)
    weight_masked = mask_neg * (weight / (weight * mask_neg + eps).sum(dim=1).unsqueeze(1))

    # if label_debug is not None:
    #   anchor_id = 0
    #   gt_label = label_debug[anchor_id]
    #   valid_negative_idx = torch.where(weight_masked[anchor_id]!=0)[0]
    #   weights = weight_masked[anchor_id, valid_negative_idx]
    #   sorted_idx = torch.argsort(weights)
    #   sorted_weights = weights[sorted_idx]
    #   labels = label_debug[valid_negative_idx]
    #   nearest_labels = labels[sorted_idx]
    #   print(torch.mean((weight_masked!=0).sum(dim=1)[:20].float()))
    #   print(gt_label)
    #   print(nearest_labels)
    #   print(sorted_weights)
    #   ipdb.set_trace()

    T = torch.exp(sim/self.tau)
    T_masked = mask_neg * ((T * mask_neg).sum(dim=1).unsqueeze(1) / T)
    weighted = (T_masked * weight_masked).sum(dim=1).unsqueeze(1)
    correction = (1 - self.clstau) * weighted
    denominator = Sv * (1 - correction) + neg
    return denominator

  def _get_chkval(self, Sv, neg, sim, mask_pos, mask_neg):
    denominator = self._get_denominator(Sv, neg, sim, mask_neg)
    # get the sorted negative indices, from most to least negative
    sim_tmp = sim.clone()
    sim_tmp[mask_neg==0] = np.inf
    sorted_neg_idx = torch.argsort(sim_tmp, dim=1)
    chk_value = mask_pos * denominator
    chk_value = chk_value - chk_value.diag().diag()
    chk_value = chk_value.sum(dim=1)
    # chk_value = denominator[mask_pos==1]
    return chk_value, sorted_neg_idx

  def _get_new_maskneg(self, Sv1, neg1, sim1, mask_pos, mask_neg):
    Sv = Sv1.detach().clone()
    neg = neg1.detach().clone()
    sim = sim1.detach().clone()
    chk_value, sorted_neg_idx = self._get_chkval(Sv, neg, sim, mask_pos, mask_neg)
    while (chk_value < 0).sum() > 0:
      # trim most negatives negatives, mask_neg[.,:n_trim)
      trim_idx = sorted_neg_idx[chk_value < 0, :self.n_trim]
      trimmed_mask = mask_neg[torch.where(chk_value < 0)[0]]
      trimmed_mask = trimmed_mask.scatter_(1, trim_idx, 0)
      mask_neg[chk_value < 0] = trimmed_mask
      chk_value, sorted_neg_idx = self._get_chkval(Sv, neg, sim, mask_pos, mask_neg)
    return mask_neg

  def hpm_contrastive_loss_v1(self, z, mask_pos, mask_neg, n_s, label_debug=None):
    # normalize
    z = F.normalize(z, dim=1)
    bsz, featdim = z.size()
    z_square = z.view(bsz, 1, featdim).repeat(1, bsz, 1)
    sim = nn.CosineSimilarity(dim=2)(z_square, z_square.transpose(1, 0))
    Sv = torch.exp(sim / self.tau)
    neg = (Sv.clone()*mask_neg.clone()).sum(dim=1).unsqueeze(1).repeat(1,bsz)
    numerator = Sv
    mask_neg = self._get_new_maskneg(Sv, neg, sim, mask_pos, mask_neg)
    if label_debug is not None:
      self._get_denominator(Sv, neg, sim, mask_neg, label_debug=label_debug)
    denominator = self._get_denominator(Sv, neg, sim, mask_neg)
    mask_pos = mask_pos - mask_pos.diag().diag()
    if torch.isnan(torch.log(numerator/(denominator+eps)+eps)[mask_pos==1]).sum()>0:
      print('FOUND NAN VALUES!!')
      ipdb.set_trace()
    else:
      li = torch.log(numerator / (denominator + eps) + eps)[mask_pos == 1]
    loss = -li.mean(dim=0)
    return loss

  def hpm_contrastive_loss_v2(self, z, mask_pos, mask_neg, n_s):
    # normalize
    z = F.normalize(z, dim=1)
    bsz, featdim = z.size()
    z_square = z.view(bsz, 1, featdim).repeat(1, bsz, 1)
    sim = nn.CosineSimilarity(dim=2)(z_square, z_square.transpose(1, 0))
    Sv = torch.exp(sim / self.tau)
    neg = (Sv * mask_neg).sum(dim=1).unsqueeze(1).repeat(1, bsz)
    li = mask_pos * torch.log(Sv / (Sv + neg) + eps)
    li = li - li.diag().diag()
    li = (1 / (n_s - 1)) * li.sum(dim=1)
    # Note: dividing li.sum by mask.sum takes care for different strategies of incuding paired-positives in the loss
    # in the vanila case (simclr), the operation boils down to li.mean
    active_anchors = mask_pos.sum(dim=1) > 0
    loss_simclr = -li.sum(dim=0) / active_anchors.sum(dim=0)

    # hpm_reg
    ## original formulation
    # T = torch.log(Sv/(Sv+neg))
    # modified
    T = torch.log(Sv / (neg+eps))
    weight = torch.exp(-self.beta * sim) * mask_neg
    weight = weight / (weight.sum(dim=1).unsqueeze(1) + eps)
    reg_i = (weight * T).sum(dim=1)
    active_anchors = mask_pos.sum(dim=1) > 0
    loss_hpm = reg_i.sum(dim=0) / active_anchors.sum(dim=0)

    loss = loss_simclr + (1-self.clstau) * loss_hpm
    return loss


  def LPAN_contrastive_loss(self, x, shots_per_way, n_way, unlab_bsz, mode='standard', label_debug=None):
    # labelled positives and all negatives
    bsz,n_s = x.size()[:2]
    n_l = n_way * shots_per_way
    n_ul = unlab_bsz * n_s
    x = x.cuda()
    x = x.contiguous().view(bsz * n_s, *x.size()[2:])
    z, _ = self.forward_this(x)

    # positive mask
    T1 = np.eye(int(n_l/n_s))
    T2 = np.ones((n_s, n_s))
    mask_pos_lab = torch.FloatTensor(np.kron(T1, T2))
    T3 = torch.cat([mask_pos_lab, torch.zeros(n_l, n_ul)], dim=1)
    T4 = torch.cat([torch.zeros(n_ul, n_l), torch.zeros(n_ul,n_ul)], dim=1)
    mask_pos = torch.cat([T3,T4], dim=0).to(z.device)

    # negative mask
    T1 = 1-np.eye(n_way)
    T2 = np.ones((shots_per_way, shots_per_way))
    mask_neg_lab = torch.FloatTensor(np.kron(T1, T2))
    T3 = torch.cat([mask_neg_lab, torch.ones(n_l, n_ul)], dim=1)
    T4 = torch.cat([torch.zeros(n_ul, n_l), torch.zeros(n_ul,n_ul)], dim=1)
    mask_neg = torch.cat([T3,T4], dim=0).to(z.device)
    if mode=='standard':
      return self.general_contrastive_loss(z, mask_pos, mask_neg, n_s)
    elif mode=='hpm_v1':
      # return self.hpm_contrastive_loss_v1(z, mask_pos, mask_neg, n_s, label_debug)
      return self.hpm_contrastive_loss_v1(z, mask_pos, mask_neg, n_s)
    elif mode=='hpm_v2':
      return self.hpm_contrastive_loss_v2(z, mask_pos, mask_neg, n_s)


  def LPAN_GroundTruth(self, x, y_lab, y_unlab, shots_per_way, n_way, unlab_bsz, mode='standard'):
    # labelled positives and all negatives
    bsz,n_s = x.size()[:2]
    n_l = n_way * shots_per_way
    n_ul = unlab_bsz * n_s
    x = x.cuda()
    x = x.contiguous().view(bsz * n_s, *x.size()[2:])
    z, _ = self.forward_this(x)

    # positive mask
    T1 = np.eye(int(n_l/n_s))
    T2 = np.ones((n_s, n_s))
    mask_pos_lab = torch.FloatTensor(np.kron(T1, T2))
    T3 = torch.cat([mask_pos_lab, torch.zeros(n_l, n_ul)], dim=1)
    T4 = torch.cat([torch.zeros(n_ul, n_l), torch.zeros(n_ul,n_ul)], dim=1)
    mask_pos = torch.cat([T3,T4], dim=0).to(z.device)

    # negative mask
    y = torch.cat([y_lab,y_unlab]).view(1,-1).repeat(n_l,1)
    label_comp = y_lab.view(-1,1).repeat(1,n_l + n_ul)
    T3 = (y!=label_comp).float()
    T4 = torch.ones(n_ul, n_l+n_ul) #dummy
    mask_neg = torch.cat([T3,T4], dim=0).to(z.device)
    if mode=='standard':
      return self.general_contrastive_loss(z, mask_pos, mask_neg, n_s)
    elif mode=='hpm_v1':
      return self.hpm_contrastive_loss_v1(z, mask_pos, mask_neg, n_s)
    elif mode=='hpm_v2':
      return self.hpm_contrastive_loss_v2(z, mask_pos, mask_neg, n_s)


  def OL_contrastive_loss(self, x, shots_per_way):
    bsz,n_s = x.size()[:2]
    x = x.cuda()
    x = x.contiguous().view(bsz * n_s, *x.size()[2:])
    z, _ = self.forward_this(x)
    # positive mask
    T1 = np.eye(bsz)
    T2 = np.ones((n_s, n_s))
    mask_pos = torch.FloatTensor(np.kron(T1, T2)).to(z.device)
    # negative mask
    n_way = int((bsz*n_s)/shots_per_way)
    T1 = 1-np.eye(n_way)
    T2 = np.ones((shots_per_way, shots_per_way))
    mask_neg = torch.FloatTensor(np.kron(T1, T2)).to(z.device)
    return self.general_contrastive_loss(z, mask_pos, mask_neg, n_s)

  def Apan_contrastive_loss(self, x, shots_per_way, n_way, unlab_bsz):
    # all positives and all negatives
    bsz,n_s = x.size()[:2]
    n_l = n_way * shots_per_way
    n_ul = unlab_bsz * n_s
    x = x.cuda()
    x = x.contiguous().view(bsz * n_s, *x.size()[2:])
    z, _ = self.forward_this(x)

    # positive mask
    T1p = np.eye(bsz)
    T2p = np.ones((n_s, n_s))
    mask_pos = torch.FloatTensor(np.kron(T1p, T2p)).to(z.device)

    # negative mask
    T1n = 1-np.eye(n_way)
    T2n = np.ones((shots_per_way, shots_per_way))
    mask_neg_lab = torch.FloatTensor(np.kron(T1n, T2n))
    T3 = torch.cat([mask_neg_lab, torch.ones(n_l, n_ul)], dim=1)
    T41 = 1-np.eye(unlab_bsz)
    mask_neg_Unlab = torch.FloatTensor(np.kron(T41, T2p))
    T4 = torch.cat([torch.ones(n_ul, n_l), mask_neg_Unlab], dim=1)
    mask_neg = torch.cat([T3,T4], dim=0).to(z.device)
    return self.general_contrastive_loss(z, mask_pos, mask_neg, n_s)
