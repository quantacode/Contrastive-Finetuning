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
from torch.nn.utils import weight_norm
import ipdb
eps=0.00001
class CE_model(SIMCLR):
  def __init__(self, model_func, tau, n_way, tf_path, loadpath=None, is_distribute=False,
               beta=0, clstau=0, n_trim=1):
    super(CE_model, self).__init__(model_func=model_func,
                               tau=tau, n_way=n_way,
                               tf_path=tf_path, loadpath=loadpath,
                               is_distribute=is_distribute)
    # self.classifier = nn.Linear(self.feature.final_feat_dim, n_way)
    self.L = weight_norm(nn.Linear(self.feature.final_feat_dim, n_way, bias=False), name='weight', dim=0)
    self.relu = nn.ReLU()
    self.loss_fn = nn.CrossEntropyLoss().cuda()
    self.method = 'CSSF_classifier'

  def forward_classifier(self, x):
    x = self.feature(x)
    x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
    x_normalized = x.div(x_norm + eps)
    L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
    self.L.weight.data = self.L.weight.data.div(L_norm + eps)
    cos_dist = self.L(x_normalized)
    scores = 10 * cos_dist
    return scores

  def CE_loss(self, x, y):
    out = self.forward_classifier(x)
    loss = self.loss_fn(out, y)
    return loss