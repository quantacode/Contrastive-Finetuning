# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
import random
import ipdb
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

class NamedDataset(SimpleDataset):
  def __init__(self, data_file, transform, target_transform=identity):
    super(NamedDataset, self).__init__(data_file, transform, target_transform=target_transform)

  def __getitem__(self, i):
    img_name = self.meta['image_names'][i]
    image_path = os.path.join(img_name)
    img = Image.open(image_path).convert('RGB')
    img = self.transform(img)
    target = self.target_transform(self.meta['image_labels'][i])
    return img_name, img, target


class SetDataset:
  def __init__(self, data_file, batch_size, transform):
    with open(data_file, 'r') as f:
      self.meta = json.load(f)
    self.cl_list = np.unique(self.meta['image_labels']).tolist()
    self.sub_meta = {}
    for cl in self.cl_list:
      self.sub_meta[cl] = []
    for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
      self.sub_meta[y].append(x)
    self.sub_dataloader = []
    sub_data_loader_params = dict(batch_size = batch_size,
        shuffle = True,
        num_workers = 0, #use main thread only or may receive multiple batches
        pin_memory = False)
    for cl in self.cl_list:
      sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform )
      self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

  def __getitem__(self,i):
    return next(iter(self.sub_dataloader[i]))

  def __len__(self):
    return len(self.cl_list)


class SubDataset:
  def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity, min_size=50):
    self.sub_meta = sub_meta
    self.cl = cl
    self.transform = transform
    self.target_transform = target_transform
    if len(self.sub_meta) < min_size:
      idxs = [i % len(self.sub_meta) for i in range(min_size)]
      self.sub_meta = np.array(self.sub_meta)[idxs].tolist()

  def __getitem__(self,i):
    image_path = os.path.join( self.sub_meta[i])
    img = Image.open(image_path).convert('RGB')
    # if 'aircraft/images' in image_path:
    #   img = img.crop((0, 0, img.size[0], img.size[1] - 20))
    img = self.transform(img)
    target = self.target_transform(self.cl)
    return img, target

  def __len__(self):
    return len(self.sub_meta)

##############
class MultiSetDataset:
  def __init__(self, data_files, batch_size, transform):
    self.cl_list = np.array([])
    self.sub_dataloader = []
    self.n_classes = []
    for data_file in data_files:
      with open(data_file, 'r') as f:
        meta = json.load(f)
      cl_list = np.unique(meta['image_labels']).tolist()
      self.cl_list = np.concatenate((self.cl_list, cl_list))

      sub_meta = {}
      for cl in cl_list:
        sub_meta[cl] = []

      for x,y in zip(meta['image_names'], meta['image_labels']):
        sub_meta[y].append(x)

      sub_data_loader_params = dict(batch_size = batch_size,
          shuffle = True,
          num_workers = 0, #use main thread only or may receive multiple batches
          pin_memory = False)
      for cl in cl_list:
        sub_dataset = SubDataset(sub_meta[cl], cl, transform = transform, min_size=batch_size)
        self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )
      self.n_classes.append(len(cl_list))

  def __getitem__(self,i):
    return next(iter(self.sub_dataloader[i]))

  def __len__(self):
    return len(self.cl_list)

  def lens(self):
    return self.n_classes


class SetDataset_AugEpisode:
  def __init__(self, data_file, batch_size, support_size, tx):
    self.batch_size = batch_size
    self.support_size = support_size
    self.transform_support = tx[0]
    self.transform_query = tx[1]
    with open(data_file, 'r') as f:
      meta = json.load(f)
    self.impaths = meta['image_names']

  def __getitem__(self,i):
    image_path = os.path.join(self.impaths[i])
    img_raw = Image.open(image_path).convert('RGB')
    images = torch.FloatTensor([])
    labels = torch.LongTensor([])
    # duplicate each sample i n_support times without aug.
    for b in range(self.batch_size):
      if b<self.support_size:
        img = self.transform_support(img_raw)
        target = support_label
      else:
        img = self.transform_query(img_raw)
        target = query_label
      images = torch.cat([images, img.unsqueeze(0)], dim=0)
      labels = torch.cat([labels, torch.LongTensor([target])], dim=0)
    return images, labels

  def __len__(self):
    return len(self.impaths)

#########################################################################
class SubDataset:
  def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity, min_size=50):
    self.sub_meta = sub_meta
    self.cl = cl
    self.transform = transform
    self.target_transform = target_transform
    if len(self.sub_meta) < min_size:
      idxs = [i % len(self.sub_meta) for i in range(min_size)]
      self.sub_meta = np.array(self.sub_meta)[idxs].tolist()

  def __getitem__(self,i):
    image_path = os.path.join( self.sub_meta[i])
    img = Image.open(image_path).convert('RGB')
    # if 'aircraft/images' in image_path:
    #   img = img.crop((0, 0, img.size[0], img.size[1] - 20))
    img = self.transform(img)
    target = self.target_transform(self.cl)
    return img, target

  def __len__(self):
    return len(self.sub_meta)


class EpisodicBatchSampler(object):
  def __init__(self, n_classes, n_way, n_episodes):
    self.n_classes = n_classes
    self.n_way = n_way
    self.n_episodes = n_episodes

  def __len__(self):
    return self.n_episodes

  def __iter__(self):
    for i in range(self.n_episodes):
      yield torch.randperm(self.n_classes)[:self.n_way]


class MultiEpisodicBatchSampler(object):
  def __init__(self, n_classes, n_way, n_episodes):
    self.n_classes = n_classes
    self.n_way = n_way
    self.n_episodes = n_episodes
    self.n_domains = len(n_classes)

  def __len__(self):
    return self.n_episodes

  def __iter__(self):
    domain_list = [i%self.n_domains for i in range(self.n_episodes)]
    random.shuffle(domain_list)
    for i in range(self.n_episodes):
      domain_idx = domain_list[i]
      start_idx = sum(self.n_classes[:domain_idx])
      yield torch.randperm(self.n_classes[domain_idx])[:self.n_way] + start_idx


class EpisodicBatchSampler_Unsuperv(object):
  def __init__(self, n_samples, n_way, n_episodes):
    self.n_samples = n_samples
    self.n_way = n_way
    self.n_episodes = n_episodes

  def __len__(self):
    return self.n_episodes

  def __iter__(self):
    for i in range(self.n_episodes):
      samples = torch.randperm(self.n_samples)[:self.n_way]
      yield samples
