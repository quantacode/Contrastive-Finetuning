# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import json
import numpy as np
import os
import random
from PIL import Image
import ipdb

import torch
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from abc import abstractmethod

NUM_WORKERS=2
class TransformLoader:
  def __init__(self, image_size,
      normalize_param = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      jitter_param = dict(Brightness=0.4, Contrast=0.4, Color=0.4), rot_deg = 30):
    self.image_size = image_size
    self.normalize_param = normalize_param
    self.jitter_param = jitter_param
    self.rot_deg = rot_deg

  def parse_transform(self, transform_type):
    if transform_type=='ImageJitter':
      method = add_transforms.ImageJitter( self.jitter_param )
      return method
    method = getattr(transforms, transform_type)

    if transform_type=='Grayscale':
      return method(3)
    elif transform_type=='RandomResizedCrop':
      # return method(self.image_size)
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
      transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
    else:
      transform_list = ['Resize','CenterCrop', 'ToTensor', 'Normalize']

    transform_funcs = []
    for x in transform_list:
      transform_funcs.append(self.parse_transform(x))
    transform = transforms.Compose(transform_funcs)
    return transform

#class DataManager:
class DataManager(object):
  @abstractmethod
  def get_data_loader(self, data_file, aug):
    pass

class SimpleDataManager(DataManager):
  def __init__(self, image_size, batch_size, drop_last=False, is_shuffle=True):
    super(SimpleDataManager, self).__init__()
    self.batch_size = batch_size
    self.trans_loader = TransformLoader(image_size)
    self.drop_last = drop_last
    self.is_shuffle = is_shuffle

  def get_data_loader(self, data_file, aug):
    transform = self.trans_loader.get_composed_transform(aug)
    dataset = SimpleDataset(data_file, transform)
    data_loader_params = dict(batch_size = self.batch_size, shuffle = self.is_shuffle,
                              num_workers = NUM_WORKERS, pin_memory =True, drop_last=self.drop_last)
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
    return data_loader

#############

identity = lambda x:x

support_label = 1
query_label = 0
class SimpleDataset:
  def __init__(self, data_file, transform, target_transform=identity):
    with open(data_file, 'r') as f:
      self.meta = json.load(f)

    classnames_filename = os.path.join(os.path.dirname(data_file), 'classnames.txt')
    if os.path.exists(classnames_filename):
      self.clsid2name = {}
      with open(classnames_filename) as f:
        lines = f.readlines()
        for line in lines:
          line = line.split('\n')[0]
          if '#' not in line and line!='':
            try:
              clsid, clsname = line.split(' ')
            except:
              ipdb.set_trace()
            self.clsid2name[clsid] = clsname

    self.transform = transform
    self.target_transform = target_transform

  def __getitem__(self,i):
    image_path = os.path.join(self.meta['image_names'][i])
    img = Image.open(image_path).convert('RGB')
    img = self.transform(img)
    target = self.target_transform(self.meta['image_labels'][i])
    return img, target

  def __len__(self):
    return len(self.meta['image_names'])

  def get_classname(self, filename):
    clsid = os.path.dirname(filename).split('/')[-1]
    return self.clsid2name[clsid]