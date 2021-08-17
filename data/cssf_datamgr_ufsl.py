# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from data.dataset import SetDataset,EpisodicBatchSampler
from data.datamgr_ufsl import Omniglot_TransformLoader, Miniimagenet_TransformLoader, Celeba_TransformLoader
from abc import abstractmethod
import json
import numpy as np
import os
from PIL import Image
import ipdb
import matplotlib.pyplot as plt
import torch.nn.functional as F

# novel
NW_setDM = 0
NW_labCB = 0
# source
NW_simpleDM = 0
NW_unlabCB = 0
identity = lambda x:x

class DataManager:
  @abstractmethod
  def get_data_loader(self, data_file, aug):
    pass

##############################################################################
class Augmentator:
  def __init__(self, x, transform, y=None):
    self.x = []
    for xi in x:
      self.x.extend(xi)
    self.transform = transform
    if y is not None:
      self.y = y.contiguous().view(-1)
    else:
      self.y = ['None']*len(self.x)

  def __len__(self):
    return len(self.x)

  def __getitem__(self, i):
    img = self.transform(self.x[i])
    target = self.y[i]
    return img, target

class UnsupCon_Augmentator (Augmentator):
  # for 1 shot experiments
  def __init__(self, x, transform):
    super(UnsupCon_Augmentator, self).__init__(x, transform)

  def __getitem__(self, i):
    img1 = self.transform(self.x[i])
    img2 = self.transform(self.x[i])
    img = torch.cat([img1.unsqueeze(0), img2.unsqueeze(0)], dim=0)
    target = self.y[i]
    return img, target

class ContrastiveBatchifier:
  def __init__(self, n_way, n_support, image_size, dataset):
    self.n_way=n_way
    self.n_support=n_support
    if dataset == 'omniglot':
      self.transform = Omniglot_TransformLoader(image_size).get_composed_transform(aug=True)
    elif dataset == 'miniImagenet':
      self.transform = Miniimagenet_TransformLoader(image_size).get_composed_transform(aug=True)
    elif dataset == 'celeba':
      self.transform = Celeba_TransformLoader(image_size).get_composed_transform(aug=True)

  def get_loader(self, x):
    if self.n_support>1:
      dataset = Augmentator(x, self.transform)
    else:
      dataset = UnsupCon_Augmentator(x, self.transform)
    data_loader_params = dict(
      dataset=dataset,
      batch_size = len(dataset),
      shuffle = False,# batchify fn below expects this to be False
      num_workers = NW_labCB,
      pin_memory = True)
    data_loader = torch.utils.data.DataLoader(**data_loader_params)
    return data_loader

  def _batchify(self, x):
    # converts into contrastive batch
    x = x.contiguous().view(self.n_way, self.n_support,*x.shape[1:])
    permuted_idx = torch.cat([torch.randperm(self.n_support).unsqueeze(0) for _ in range(self.n_way)], dim=0)
    shots_per_way = self.n_support if self.n_support % 2 == 0 else self.n_support - 1
    permuted_idx = permuted_idx[:, :shots_per_way]
    permuted_idx = permuted_idx.view(self.n_way, shots_per_way, 1, 1, 1).expand(
      self.n_way, shots_per_way, *x.shape[2:]).to(x.device)
    x = torch.gather(x, 1, permuted_idx)
    x = x.view(-1, *x.shape[2:])
    bch = torch.split(x, 2)
    bch = torch.cat([b.unsqueeze(0) for b in bch], dim=0)
    return bch, shots_per_way

  def batchify(self, x):
    if self.n_support>1:
      return self._batchify(x)
    else:
      return x, 2

if __name__=="__main__":
  x = torch.rand(5,5,3,224,224)
  Contrastive_batchify(x)
