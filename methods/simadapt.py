import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from methods.gnnnet import GnnNet
from methods import backbone
import ipdb

class SimAda(GnnNet):
  maml=False
  def __init__(self, model_func,  n_way, n_support, num_clusters=100, sim_bias=0.0, tf_path=None, loadpath=None):
    super(SimAda, self).__init__(model_func,  n_way, n_support, tf_path=tf_path)

    # loss function
    self.loss_cluster = nn.BCEWithLogitsLoss(reduction='none')

    # ~ params for cluster learning
    # self.sim_bias = nn.Parameter(torch.rand(1))
    self.sim_bias = sim_bias
    self.cluster_classifier = nn.Linear(128, num_clusters)

    if loadpath!=None:
      self.load_simada(loadpath)

    if (torch.cuda.device_count() > 1):
      # self.modules = nn.DataParallel(self.modules)
      self.cluster_classifier =nn.DataParallel(self.cluster_classifier )

    self.method = 'SimAda'
    # fix label for training the metric function   1*nw(1 + ns)*nw
    support_label = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).unsqueeze(1)
    support_label = torch.zeros(self.n_way*self.n_support, self.n_way).scatter(1, support_label, 1).view(self.n_way, self.n_support, self.n_way)
    support_label = torch.cat([support_label, torch.zeros(self.n_way, 1, n_way)], dim=1)
    self.support_label = support_label.view(1, -1, self.n_way)

  def cuda(self):
    self.feature.cuda()
    self.fc.cuda()
    self.gnn.cuda()
    self.cluster_classifier.cuda()
    self.support_label = self.support_label.cuda()
    return self

  def load_simada(self, loadpath):
    state = torch.load(loadpath)['state']
    self.load_state_dict(state, strict=False)
    return self

  def cluster_rep_params(self):
    """
	This generator returns parameters for feature, cluster classifier, fc (projection layer)
	"""
    b = []

    b.append(self.feature)
    b.append(self.fc)
    b.append(self.cluster_classifier)

    for i in range(len(b)):
      for j in b[i].modules():
        jj = 0
        for k in j.parameters():
          jj += 1
          if k.requires_grad:
            yield k

  def cluster_sim_params(self):
    """
    This generator returns parameters for  gnn layer
	"""
    b = []

    b.append(self.gnn)

    for i in range(len(b)):
      for j in b[i].modules():
        jj = 0
        for k in j.parameters():
          jj += 1
          if k.requires_grad:
            yield k

  def clusterReg(self, x):
    bsz = x.shape[0]
    z_all = self.fc(self.feature(x))
    prob = F.softmax(self.cluster_classifier(z_all), dim=-1)
    pleft = prob.unsqueeze(1).repeat(1,bsz,1)
    pleft = pleft.view(bsz * bsz, -1)
    pright = prob.unsqueeze(0).repeat(bsz,1,1)
    pright = pright.view(bsz * bsz, -1)
    logits = pleft.unsqueeze(1).bmm(pright.unsqueeze(2)).squeeze()

    # groupify the matrix computation to fit into the memory
    grp_sz = 32
    num_grps = int(bsz/grp_sz)
    S_all = torch.zeros(bsz, bsz).to(z_all.device)
    for rowidx in range(num_grps):
      z = z_all[rowidx * grp_sz : (rowidx+1) * grp_sz, :].unsqueeze(1)
      for colidx in range(rowidx, num_grps):
        zt = z_all[colidx * grp_sz : (colidx+1) * grp_sz, :].unsqueeze(1)
        zt = torch.transpose(zt, 1, 0)  # gsize: bs x N x N x num_features

        delta = torch.abs(z - zt).unsqueeze(0)  # size: bs x N x N x num_features
        delta = torch.transpose(delta, 1, 3)  # size: bs x num_features x N x N
        if (torch.cuda.device_count() > 1):
          similarity_model = self.gnn.module.module_w
        else:
          similarity_model = self.gnn.module_w
        S = torch.exp(similarity_model._forward(delta)).squeeze(0).squeeze(2)
        S_all[rowidx * grp_sz : (rowidx+1) * grp_sz, colidx * grp_sz : (colidx+1) * grp_sz] = S.clone()
        S_all[colidx * grp_sz : (colidx+1) * grp_sz, rowidx * grp_sz : (rowidx+1) * grp_sz] = S.clone().transpose(1,0)

    # labels = torch.sigmoid(S_all)
    labels = S_all/S_all.max()
    labels = labels.view(bsz * bsz)
    mask = (labels>self.sim_bias).to(labels.device)
    loss = self.loss_cluster(logits*mask, labels.cuda()*mask)
    loss = loss.view(bsz, bsz)
    loss = loss[torch.triu_indices(loss.shape[0], loss.shape[1], offset=1)]
    return loss.mean()
