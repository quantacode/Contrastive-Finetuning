# This code is modified from https://github.com/jakesnell/prototypical-networks

import torch
import torch.nn as nn
import numpy as np
from methods.meta_template import MetaTemplate
import torch.nn.functional as F
import ipdb
from utils import chkpt_vis


class ProtoNet_cosine(MetaTemplate):
  def __init__(self, model_func,  n_way, n_support, tau=0.5, tf_path=None, loadpath=None, is_distribute=False):
    super(ProtoNet_cosine, self).__init__(model_func,  n_way, n_support, tf_path=tf_path)
    self.loss_fn = nn.CrossEntropyLoss()
    self.tau = tau
    # self.feature = nn.Sequential(
    #   self.feature,
    #   nn.Linear(self.feature.final_feat_dim, 128),
    #   # nn.ReLU(),
    #   # nn.Linear(128, 64)
    # )
    self.method = 'ProtoNet_cosine'
    if loadpath!=None:
      self.load_protonet(loadpath)
    if is_distribute:
      self.distribute_model()


  def load_protonet(self, loadpath):
    state = torch.load(loadpath)
    loadstate = {}
    if 'model' in state:
      if 'ResNet10' in loadpath:
        # resnet10 model load from Supcon paper
        state = state['model']
        for key in state.keys():
          if 'encoder.' in key:
            loadstate[key.replace('encoder.', 'feature.')] = state[key]
          else:
            loadstate[key] = state[key]
      else:
        state = state['model']
        for key in self.state_dict().keys():
          if key.replace('feature.','') in state.keys():
            # resnet12 model load from rfs paper
            loadstate[key] = state[key.replace('feature.','')]
          elif key.replace('feature.','encoder.') in state.keys():
            # resnet12_scl
            loadstate[key] = state[key.replace('feature.','encoder.')]
    elif 'state_dict' in state.keys():
      state = state['state_dict']
      for key in self.state_dict().keys():
        if key.replace('feature.', '') in state.keys():
          # resnet18_cnaps
          loadstate[key] = state[key.replace('feature.', '')]
    elif 'state' in state.keys():
      state = state['state']
      for key in state.keys():
        if 'feature.module' in key:
          loadstate[key.replace('feature.module', 'feature')] = state[key]
        else:
          loadstate[key] = state[key]
    else:
      raise ValueError

    # chkpt_vis(self, -10, -1)
    self.load_state_dict(loadstate, strict=False)
    # chkpt_vis(self, -10, -1)
    # ipdb.set_trace()
    return self

  def resume_best_model(self, loadpath):
    loadfile = torch.load(loadpath)
    state = loadfile['state']
    self.load_state_dict(state, strict=False)
    resume_bch = loadfile['bch']
    return resume_bch

  def distribute_model(self):
      self.feature = nn.DataParallel(self.feature)

  def reset_modules(self):
    return

  def parse_feature(self, x, is_feature):
    x = x.cuda()
    if is_feature:
      z_all = x
    else:
      x           = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:])
      z_all       = self.feature(x)
      z_all       = z_all.view( self.n_way, self.n_support + self.n_query, -1)
    z_all = F.normalize(z_all, dim=2)
    z_support   = z_all[:, :self.n_support]
    z_query     = z_all[:, self.n_support:]

    return z_support, z_query

  def set_forward(self,x,is_feature=False):
    z_support, z_query  = self.parse_feature(x,is_feature)
    z_support   = z_support.contiguous()
    z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
    z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

    scores = cosine_dist(z_query, z_proto)
    return scores

  def set_forward_loss(self, x):
    y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
    y_query = y_query.cuda()
    scores = self.set_forward(x)
    loss = self.loss_fn(scores, y_query)
    return scores, loss

def cosine_dist( x, y):
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
