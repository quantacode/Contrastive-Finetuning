# contrastive semi supervised model
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from methods.weight_imprint_based import WeightImprint
from tqdm import tqdm
from scipy import stats
import random
from torch.nn.utils import weight_norm
import ipdb

EPS=0.00001
class cosine_LC(WeightImprint):
  def __init__(self, model_func, n_way, tau=0.5,  tf_path=None, loadpath=None, is_distribute=False,
               is_MT=False, src_classes=0, classifier_indim=None):
    super(cosine_LC, self).__init__(model_func=model_func, tf_path=tf_path)
    self.n_way = n_way
    self.src_classes = src_classes
    self.classifier_indim = classifier_indim
    if self.classifier_indim == None:
      self.classifier_indim = self.feature.final_feat_dim
    print('--- Classifier indim : %d ---' % (self.classifier_indim))
    self.L = weight_norm(nn.Linear(self.classifier_indim, self.n_way, bias=False), name='weight', dim=0)
    self.is_MT = is_MT
    if self.is_MT:
      self.source_L = weight_norm(nn.Linear(self.classifier_indim, src_classes, bias=False), name='weight', dim=0)
    self.loss_fn = nn.CrossEntropyLoss()
    self.method = 'CE_classifier'
    if loadpath != None:
      self.load_model(loadpath)
    if is_distribute:
      self.distribute_model()

  def refresh_from_chkpt(self):
    self.load_state_dict(self.init_checkpoint, strict=False)
    self.L = weight_norm(nn.Linear(self.classifier_indim, self.n_way, bias=False), name='weight', dim=0)
    if self.is_MT:
      self.source_L = weight_norm(nn.Linear(self.classifier_indim, self.src_classes, bias=False), name='weight',
                                  dim=0)
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

  def get_classification_scores(self, x, classifier):
    z = self.get_feature(x)
    z_norm = torch.norm(z, p=2, dim=1).unsqueeze(1).expand_as(z)
    z_normalized = z.div(z_norm + EPS)
    L_norm = torch.norm(classifier.weight.data, p=2, dim=1).unsqueeze(1).expand_as(classifier.weight.data)
    classifier.weight.data = classifier.weight.data.div(L_norm + EPS)
    cos_dist = classifier(z_normalized)
    scores = 10 * cos_dist
    return scores

  def CE_loss(self, x, y):
    scores = self.get_classification_scores(x, self.L)
    loss = self.loss_fn(scores, y)
    return loss

  def CE_loss_source(self, x, y):
    scores = self.get_classification_scores(x, self.source_L)
    loss = self.loss_fn(scores, y)
    return loss
  
########## std classifier
class LC(WeightImprint):
  def __init__(self, model_func, n_way, tau=0.5,  tf_path=None, loadpath=None, is_distribute=False, is_MT=False,
               src_classes=0):
    super(LC, self).__init__(model_func=model_func, tf_path=tf_path)
    self.n_way = n_way
    self.src_classes = src_classes
    self.is_MT = is_MT
    self.L = nn.Linear(self.feature.final_feat_dim, n_way)
    if is_MT:
      self.source_L = nn.Linear(self.feature.final_feat_dim, src_classes)
    self.loss_fn = nn.CrossEntropyLoss().cuda()
    self.method = 'CE_classifier'
    if loadpath != None:
      self.load_model(loadpath)
    if is_distribute:
      self.distribute_model()

  def refresh_from_chkpt(self):
    self.load_state_dict(self.init_checkpoint, strict=False)
    self.L = weight_norm(nn.Linear(self.feature.final_feat_dim, self.n_way, bias=False), name='weight', dim=0)
    if self.s_MT:
      self.source_L = nn.Linear(self.feature.final_feat_dim, self.src_classes)
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

  def get_classification_scores(self, x, classifier):
    z = self.get_feature(x)
    scores = classifier(z)
    return scores

  def CE_loss(self, x, y):
    scores = self.get_classification_scores(x, self.L)
    loss = self.loss_fn(scores, y)
    return loss

  def CE_loss_source(self, x, y):
    scores = self.get_classification_scores(x, self.source_L)
    loss = self.loss_fn(scores, y)
    return loss