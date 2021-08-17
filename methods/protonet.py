# This code is modified from https://github.com/jakesnell/prototypical-networks

import torch
import torch.nn as nn
import numpy as np
from methods.meta_template import MetaTemplate
import ipdb

class ProtoNet(MetaTemplate):
  def __init__(self, model_func,  n_way, n_support, tf_path=None, loadpath=None, is_distribute=False):
    super(ProtoNet, self).__init__(model_func,  n_way, n_support, tf_path=tf_path)
    self.loss_fn = nn.CrossEntropyLoss()
    self.method = 'ProtoNet'
    self.final_feat_dim = self.feat_dim
    if loadpath!=None:
      self.load_protonet(loadpath)
    if is_distribute:
      self.distribute_model()

  def load_protonet(self, loadpath):
    state = torch.load(loadpath)
    if 'model' in state:
      # resnet12 model load from rfs paper
      state = state['model']
      loadstate = {}
      for key in self.state_dict().keys():
        if key.replace('feature.','') in state.keys():
          loadstate[key] = state[key.replace('feature.','')]
    elif 'state' in state.keys():
      state = state['state']
      loadstate = {}
      for key in state.keys():
        if 'feature.module' in key:
          loadstate[key.replace('feature.module', 'feature')] = state[key]
        else:
          loadstate[key] = state[key]

    self.load_state_dict(loadstate, strict=False)
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

  def set_forward(self,x,is_feature=False):
    z_support, z_query  = self.parse_feature(x,is_feature)
    z_support   = z_support.contiguous()
    z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
    z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

    dists = euclidean_dist(z_query, z_proto)
    scores = -dists
    return scores

  def get_distance(self,x,is_feature = False):
    z_support, z_query  = self.parse_feature(x,is_feature)
    z_support   = z_support.contiguous()
    z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
    z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )
    return euclidean_dist(z_proto, z_proto)[0, :5].cpu().numpy()

  def set_forward_loss(self, x):
    y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
    y_query = y_query.cuda()
    scores = self.set_forward(x)
    loss = self.loss_fn(scores, y_query)
    return scores, loss


def euclidean_dist( x, y):
  # x: N x D
  # y: M x D
  n = x.size(0)
  m = y.size(0)
  d = x.size(1)
  assert d == y.size(1)

  x = x.unsqueeze(1).expand(n, m, d)
  y = y.unsqueeze(0).expand(n, m, d)

  return torch.pow(x - y, 2).sum(2)
