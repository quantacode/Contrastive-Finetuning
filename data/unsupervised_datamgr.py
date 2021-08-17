# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from abc import abstractmethod
import json
import os
from PIL import Image
import ipdb
import numpy as np
from tqdm import tqdm

NUM_WORKERS=8
import ipdb

class ContrastiveWrapper(object):
  def __init__(self, transform, num_aug=15):
    self.transform = transform
    self.num_aug = num_aug

  def __call__(self, sample):
    xlist = []
    for i in range(self.num_aug):
      xlist.append(self.transform(sample))
    return xlist

class TransformLoader:
  def __init__(self, image_size,
      normalize_param = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      jitter_param = dict(Brightness=0.9, Contrast=0.9, Color=0.9), rot_deg = 45,
               is_aug_valoader = False, no_color=False):
    self.image_size = image_size
    self.normalize_param = normalize_param
    self.jitter_param = jitter_param
    self.rot_deg = rot_deg
    self.is_aug_valoader = is_aug_valoader
    self.no_color = no_color

  def parse_transform(self, transform_type):
    if transform_type=='ImageJitter':
      method = add_transforms.ImageJitter( self.jitter_param )
      return method
    method = getattr(transforms, transform_type)

    if transform_type=='Grayscale':
      return method(3)
    elif transform_type=='RandomResizedCrop':
      return method(self.image_size, scale=(0.2, 1.0))
    elif transform_type=='CenterCrop':
      return method(self.image_size)
    elif transform_type=='Resize':
      return method([int(self.image_size*1.15), int(self.image_size*1.15)])
    elif transform_type=='RandomRotation':
      return method(self.rot_deg)
    elif transform_type=='Normalize':
      return method(**self.normalize_param )
    else:
      return method()

  def get_composed_transform(self, aug = False):
    if aug:
      if self.is_aug_valoader:
        transform_list = ['RandomResizedCrop', 'AutoAugment', 'ToTensor', 'Normalize']
      elif self.no_color:
        transform_list = ['Grayscale', 'RandomResizedCrop', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
      else:
        transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
    else:
      if self.no_color:
        transform_list = ['Grayscale', 'Resize','CenterCrop', 'ToTensor', 'Normalize']
      else:
        transform_list = ['Resize','CenterCrop', 'ToTensor', 'Normalize']

    transform_funcs = []
    for x in transform_list:
      if x=='AutoAugment':
        transform_funcs.append(ImageNetPolicy())
      else:
        transform_funcs.append(self.parse_transform(x))
    transform = transforms.Compose(transform_funcs)
    return transform

class DataManager:
  @abstractmethod
  def get_data_loader(self, data_file, aug):
    pass

class UnsupSetDataManager(DataManager):
  def __init__(self, image_size, n_way, n_query, n_episode=100, no_color=False):
    super(UnsupSetDataManager, self).__init__()
    self.image_size = image_size
    self.n_way = n_way
    self.n_support = 1
    self.batch_size = self.n_support + n_query
    self.n_episode = n_episode

    self.trans_loader = TransformLoader(image_size, no_color=no_color)

  def get_data_loader(self, data_file): #parameters that would change on train/val set
    transform = self.trans_loader.get_composed_transform(True)
    dataset = UnsupSetDataset( data_file , self.batch_size, transform )
    sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode )
    data_loader_params = dict(batch_sampler = sampler,  num_workers = NUM_WORKERS)
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
    return data_loader

#############################################################################

class UnsupSetDataset:
  def __init__(self, data_file, batch_size, transform):
    self.transform = ContrastiveWrapper(transform, batch_size)
    with open(data_file, 'r') as f:
      self.meta = json.load(f)

  def __getitem__(self,i):
    image_path = os.path.join(self.meta['image_names'][i])
    img = Image.open(image_path).convert('RGB')
    img = self.transform(img)
    return torch.cat([im.unsqueeze(0) for im in img], dim=0), self.meta['image_labels'][i]

  def __len__(self):
    return len(self.meta['image_labels'])


class EpisodicBatchSampler(object):
  def __init__(self, n_samples, n_way, n_episodes):
    self.n_samples = n_samples
    self.n_way = n_way
    self.n_episodes = n_episodes

  def __len__(self):
    return self.n_episodes

  def __iter__(self):
    for i in range(self.n_episodes):
      yield torch.randperm(self.n_samples)[:self.n_way]
      
if __name__=="__main__":
  target_file = os.path.join('./filelists/CUB/practice.json')
  tgt_datamgr = UnsupSetDataManager(224, n_query=15, n_way=5, n_episode=1000)
  tgt_loader = tgt_datamgr.get_data_loader(target_file)
  count = []
  for x,y in tqdm(tgt_loader):
    count.append(len(np.unique(y))==5)
  print(np.mean(count))
