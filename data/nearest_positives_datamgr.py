# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from data.gaussian_blur import GaussianBlur
from abc import abstractmethod
import json
import os
from PIL import Image, ImageFile
import ipdb
import numpy as np
from tqdm import tqdm
#from data.autoaugment import ImageNetPolicy

NUM_WORKERS=4
ImageFile.LOAD_TRUNCATED_IMAGES = True
from methods.protonet_cosine_based.HPM_utils import get_hp_indices
import ipdb

class ContrastiveWrapper(object):
  def __init__(self, transform, num_aug=2):
    self.transform = transform
    self.num_aug = num_aug

  def __call__(self, sample):
    xlist = []
    for i in range(self.num_aug):
      xlist.append(self.transform(sample))
    return xlist


class TransformLoader:
  def __init__(self, image_size):
    self.image_size = image_size
    self.s = 1

  def get_composed_transform(self, aug, augstrength):
    if aug:
      if augstrength=='relaxed':
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(0.4 * self.s, 0.4 * self.s, 0.4 * self.s, 0.4 * self.s)
        kernel_size = int(0.1 * self.image_size)
        kernel_size += (1-kernel_size%2)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.image_size),
                                              transforms.RandomHorizontalFlip(p=0.5),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=kernel_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229,0.224, 0.225])])
      elif augstrength=='aggressive':
        color_jitter = dict(Brightness=0.9, Contrast=0.9, Color=0.9)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.image_size, scale=(0.2, 1.0)),
                                              add_transforms.ImageJitter(color_jitter),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    else:
      data_transforms = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return data_transforms

  def get_composed_transform2(self, aug=True):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    jitter_coeff = 0.1
    color_jitter = transforms.ColorJitter(jitter_coeff * self.s, jitter_coeff * self.s,
                                          0 * self.s, 0 * self.s)
    kernel_size = int(0.1 * self.image_size)
    kernel_size += (1-kernel_size%2)
    if aug:
      data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.image_size,
                                                                         scale=(0.4,0.6),
                                                                         ratio=(1,1)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    else:
      data_transforms = transforms.Compose([transforms.Resize([int(self.image_size*1.15), int(self.image_size*1.15)]),
                                            transforms.CenterCrop(self.image_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return data_transforms
#############################################################################

class NPDataset:
  def __init__(self, data_file, affinity, num_samples, transform, hpm_type, debug=False):
    self.update_affinity(affinity)
    self.num_samples = num_samples
    self.transform = transform
    self.hpm_type = hpm_type
    self.debug = debug
    # self.transform = ContrastiveWrapper(transform, num_samples+1)
    with open(data_file, 'r') as f:
      self.meta = json.load(f)

  def update_affinity(self, affinity):
    self.affinity = affinity

  def __getitem__(self,i):
    images = []
    # initialize all idx with anchor i
    all_idx = [i]
    numpos = (self.num_samples - 1)
    if self.num_samples>1:
      if self.debug:
        # k nearest positive
        clsid = self.meta['image_labels'][i]
        clsidx = np.asarray([ilab for ilab, lab in enumerate(self.meta['image_labels']) if lab==clsid])
        affinity = np.zeros(self.affinity[i].shape[0])
        affinity[clsidx] = self.affinity.mean(dim=2)[i,clsidx]
        closest_idx = list(np.argsort(affinity)[-numpos:])

        # # farthest positive
        # clsid = self.meta['image_labels'][i]
        # clsidx = np.asarray([ilab for ilab, lab in enumerate(self.meta['image_labels']) if lab == clsid])
        # affinity = np.ones(self.affinity[i].shape)*np.inf
        # affinity[clsidx] = self.affinity[i, clsidx]
        # affinity[i]=np.inf
        # closest_idx = list(np.argsort(affinity)[:numpos])
      else:
        affinity = self.affinity[i]
        closest_idx = get_hp_indices(affinity, numpos, self.hpm_type)
      all_idx.extend(closest_idx)
    for idx in all_idx:
      image_path = os.path.join(self.meta['image_names'][idx])
      img = Image.open(image_path).convert('RGB')
      img = self.transform(img)
      images.append(img.unsqueeze(0))
    target = self.meta['image_labels'][i]
    return torch.cat([im for im in images], dim=0), target

  def __len__(self):
    return len(self.meta['image_labels'])

class NamedDataset:
  def __init__(self, data_file, transform, num_aug):
    super(NamedDataset, self).__init__()
    with open(data_file, 'r') as f:
      self.meta = json.load(f)
    self.transform = ContrastiveWrapper(transform, num_aug)
  def __getitem__(self, i):
    img_name = self.meta['image_names'][i]
    image_path = os.path.join(img_name)
    img = Image.open(image_path).convert('RGB')
    img = self.transform(img)
    img = [im.unsqueeze(0) for im in img]
    img = torch.cat(img, dim=0)
    target = self.meta['image_labels'][i]
    return img_name, img, target
  def __len__(self):
    return len(self.meta['image_names'])
#############################################################################


class DataManager:
  @abstractmethod
  def get_data_loader(self, data_file, aug):
    pass

#NP
class NPDataManager(DataManager):
  def __init__(self, image_size, affinity, num_samples, batch_size, hpm_type):
    super(NPDataManager, self).__init__()
    self.image_size = image_size
    self.affinity = affinity
    self.num_samples = num_samples
    self.batch_size = batch_size
    self.hpm_type = hpm_type
    self.trans_loader = TransformLoader(image_size)

  def get_data_loader(self, data_file, shuffle=False, aug=True, augstrength='relaxed', debug='False'):
    transform = self.trans_loader.get_composed_transform(aug, augstrength)
    dataset = NPDataset(data_file, self.affinity, self.num_samples, transform, self.hpm_type, debug=debug)
    data_loader_params = dict(batch_size=self.batch_size, shuffle=shuffle,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
    return data_loader

#Named
class NamedDataManager(DataManager):
  def __init__(self, image_size, num_aug, batch_size, is_shuffle=True):
    super(NamedDataManager, self).__init__()
    self.trans_loader = TransformLoader(image_size)
    self.image_size = image_size
    self.batch_size = batch_size
    self.num_aug = num_aug
    self.is_shuffle = is_shuffle

  def get_data_loader(self, data_file, aug):  # parameters that would change on train/val set
    transform = self.trans_loader.get_composed_transform2(aug)
    dataset = NamedDataset(data_file, transform, self.num_aug)
    data_loader_params = dict(batch_size=self.batch_size, shuffle=self.is_shuffle,
                              num_workers = NUM_WORKERS, pin_memory=True)
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
    return data_loader


if __name__=="__main__":
  target_file = os.path.join('./filelists/CUB/practice.json')
  tgt_datamgr = NPDataManager(224, n_query=15, n_way=5, n_episode=1000)
  tgt_loader = tgt_datamgr.get_data_loader(target_file)
  count = []
  for x,y in tqdm(tgt_loader):
    count.append(len(np.unique(y))==5)
  print(np.mean(count))
