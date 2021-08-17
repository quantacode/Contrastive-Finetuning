# nearest positve contrastive learning
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from methods.weight_imprint_based import WeightImprint
from methods import backbone
from tqdm import tqdm
from torch.distributions.beta import Beta
from torch.distributions.uniform import Uniform
from tsnecuda import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import ipdb

EPS=0.00001
class InterpContrastiveModel(WeightImprint):
  maml = False

  def __init__(self, model_func, tau, alpha, mode='vanila', beta=1.0,
               tf_path=None, loadpath=None, is_distribute=False):
    super(InterpContrastiveModel, self).__init__(model_func=model_func, tf_path=tf_path)
    self.method = 'InterpContrastiveModel'
    self.tau = tau
    self.mode = mode

    self.contrastive_loss_fn = self.general_contrastive_loss
    self.alpha = alpha

    self.projected_feature_dim = 128
    self.projection_head = nn.Sequential(
      nn.Linear(self.feature.final_feat_dim, self.projected_feature_dim, bias=True)
    )
    if loadpath != None:
      self.load_model(loadpath)
    if is_distribute:
      self.distribute_model()

  def refresh_from_chkpt(self):
    self.load_state_dict(self.init_checkpoint, strict=False)
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

  def forward_this(self, x):
    return self.projection_head(self.get_feature(x))

  def general_contrastive_loss(self, z_anchor, z_na, mask_pos, mask_neg, n_s):
    # normalize
    bsz = z_anchor.size(0)
    z_anchor = F.normalize(z_anchor, dim=2)
    z_na = F.normalize(z_na, dim=2)
    Sv = nn.CosineSimilarity(dim=2)(z_anchor, z_na)
    Sv = torch.exp(Sv / self.tau)
    neg = (Sv * mask_neg).sum(dim=1).unsqueeze(1).repeat(1, bsz)
    li = mask_pos * torch.log(Sv / (Sv + neg) + EPS)
    li = li - li.diag().diag()
    li = (1 / (n_s - 1)) * li.sum(dim=1)
    loss = -li[mask_pos.sum(dim=1) > 0].mean()
    return loss

  def OL_contrastive_loss(self, x, shots_per_way):
    bsz,n_s = x.size()[:2]
    x = x.contiguous().view(bsz * n_s, *x.size()[2:])
    z = self.forward_this(x)

    # positive mask
    T1 = np.eye(bsz)
    T2 = np.ones((n_s, n_s))
    mask_pos = torch.FloatTensor(np.kron(T1, T2)).to(z.device)
    # negative mask
    n_way = int((bsz*n_s)/shots_per_way)
    T1 = 1-np.eye(n_way)
    T2 = np.ones((shots_per_way, shots_per_way))
    mask_neg = torch.FloatTensor(np.kron(T1, T2)).to(z.device)

    # construct few_shot graph nodes
    bsz, featdim = z.size()
    z_anchor = z.view(bsz, 1, featdim).repeat(1, bsz, 1)
    z_na = z_anchor.transpose(1,0)

    # generate classwise prototypes
    z_clswise = z.view(n_way, shots_per_way, featdim)
    # z_clswise = z.view(n_way, shots_per_way, featdim).detach()
    proto = z_clswise.mean(dim=1).unsqueeze(1).repeat(1,shots_per_way,1)
    proto = proto.view(-1,featdim)
    proto = proto.unsqueeze(1).repeat(1, bsz, 1)
    proto = proto * mask_neg.unsqueeze(2)

    # sample from beta distribution
    m = Beta(torch.tensor([10.0]), torch.tensor([self.alpha]))
    fac = m.sample(mask_neg.size()).squeeze(-1)
    fac_neg = torch.cat([fac.unsqueeze(0), 1-fac.unsqueeze(0)], dim=0).max(dim=0)[0]
    fac_neg[mask_neg==0] = 1 # no interpolation for elements in the same class
    fac_neg = fac_neg.unsqueeze(2).to(z_na.device)
    z_na = fac_neg * z_na + (1-fac_neg) * proto
    return self.contrastive_loss_fn(z_anchor, z_na, mask_pos, mask_neg, n_s)

  def LPAN_contrastive_loss(self, x, shots_per_way, n_way, n_ul):
    # labelled positives and all negatives
    n_pos = 2
    n_l = n_way * shots_per_way
    z = self.forward_this(x)

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

    # construct few_shot graph nodes
    bsz, featdim = z.size()
    z_anchor = z.view(bsz, 1, featdim).repeat(1, bsz, 1)
    z_na = z_anchor.transpose(1, 0)

    # generate classwise prototypes
    z_clswise = z[mask_pos.sum(dim=1) > 0].view(n_way, shots_per_way, featdim)
    # z_clswise = z[mask_pos.sum(dim=1) > 0].view(n_way, shots_per_way, featdim).detach()
    proto = z_clswise.mean(dim=1).unsqueeze(1).repeat(1, shots_per_way, 1)
    proto = proto.view(-1, featdim)
    proto = proto.unsqueeze(1).repeat(1, bsz, 1)
    proto = torch.cat([proto, torch.zeros(n_ul, bsz, featdim).to(proto.device)], dim=0)
    proto = proto * mask_neg.unsqueeze(2)

    # sample from beta distribution
    m = Beta(torch.tensor([10.0]), torch.tensor([self.alpha]))
    fac_neg = m.sample(mask_neg.size()).squeeze(-1)
    fac_neg[mask_neg == 0] = 1  # no interpolation for elements in the same class
    fac_neg = fac_neg.unsqueeze(2).to(z_na.device)
    z_na = fac_neg * z_na + (1 - fac_neg) * proto
    return self.contrastive_loss_fn(z_anchor, z_na, mask_pos, mask_neg, n_pos)

  def LPAN_hpm_loss(self, z, shots_per_way, n_way, n_support, n_ul):
    # z = self.projection_head(z)

    # labelled positives and all negatives
    n_pos = 2
    n_l = n_way * shots_per_way

    # positive mask
    T1 = np.eye(int(n_l/n_pos))
    T2 = np.zeros((n_pos, n_pos))
    T2[0, 1] = 1
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

    # construct few_shot graph nodes
    bsz, featdim = z.size()
    z_anchor = z.view(bsz, 1, featdim).repeat(1, bsz, 1)
    z_na = z_anchor.transpose(1, 0)

    # generate classwise prototypes
    z_clswise = z[mask_pos.sum(dim=1) > 0].view(n_way, n_support, featdim)
    # z_clswise = z[mask_pos.sum(dim=1) > 0].view(n_way, shots_per_way, featdim).detach()
    proto = z_clswise.mean(dim=1).unsqueeze(1).repeat(1, shots_per_way, 1)
    proto = proto.view(-1, featdim)
    proto = proto.unsqueeze(1).repeat(1, bsz, 1)
    proto = torch.cat([proto, torch.zeros(n_ul, bsz, featdim).to(proto.device)], dim=0)
    proto = proto * mask_neg.unsqueeze(2)

    # sample from beta distribution
    # m = Beta(torch.tensor([10.0]), torch.tensor([self.alpha]))
    m = Uniform(torch.tensor([self.alpha-0.1]), torch.tensor([self.alpha+0.1]))
    fac_neg = m.sample(mask_neg.size()).squeeze(-1)
    fac_neg[mask_neg == 0] = 1  # no interpolation for elements in the same class
    fac_neg = fac_neg.unsqueeze(2).to(z_na.device)
    z_na_interp = fac_neg * z_na + (1 - fac_neg) * proto

    # feat_before = z_na[:n_l:shots_per_way, n_l:,:]
    # feat_after = z_na_interp[:n_l:shots_per_way, n_l:,:]
    # visualise(feat_before, feat_after, z_anchor[:n_l:2,0,:].contiguous().view(n_way, n_support, -1))

    return self.contrastive_loss_fn(z_anchor, z_na_interp, mask_pos, mask_neg, n_pos)

def visualise(featU_before, featU_after, featL):
  featU_before = featU_before.cpu().detach().numpy()
  featU_after = featU_after.cpu().detach().numpy()
  featL = featL.cpu().detach().numpy()
  fig, ax = plt.subplots(featL.shape[0],1,figsize=(5,featL.shape[0]*5))
  for cls in range(featL.shape[0]):
    anchors = featL[cls]
    all_colors = np.asarray([0] * anchors.shape[0])
    all_labels = ['anchors']

    neg_before = featU_before[cls]
    all_colors = np.hstack([all_colors, np.asarray([1] * neg_before.shape[0])])
    all_labels.append('before')

    neg_after = featU_after[cls]
    all_colors = np.hstack([all_colors, np.asarray([2] * neg_after.shape[0])])
    all_labels.append('after')

    all_feat = np.vstack([anchors, neg_before, neg_after])
    all_feat = PCA(n_components=64).fit_transform(all_feat)
    tsne = TSNE(n_components=2).fit_transform(all_feat)
    scat = ax[cls].scatter(tsne[:, 0], tsne[:, 1], c=all_colors, alpha=0.5, s=10, cmap='Set1')
    lines, _ = scat.legend_elements()
    legend1 = ax[cls].legend(lines, all_labels)
    ax[cls].add_artist(legend1)
  plt.savefig('output2/general_results/interp.png')
  exit()

