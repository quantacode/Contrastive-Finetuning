import os
import torch
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod
import json
from PIL import ImageFile
from data.gaussian_blur import GaussianBlur
ImageFile.LOAD_TRUNCATED_IMAGES = True
import ipdb

import sys
sys.path.append("../")
from configs import *

NUM_WORKERS=4

class ContrastiveWrapper(object):
  def __init__(self, transform):
    self.transform = transform

  def __call__(self, sample):
    xi = self.transform(sample)
    xj = self.transform(sample)
    return xi, xj

class ContrastiveDataset:
  def __init__(self, data_file, transform):
    with open(data_file, 'r') as f:
      self.meta = json.load(f)
    self.transform = ContrastiveWrapper(transform)

  def __getitem__(self,i):
    image_path = os.path.join(self.meta['image_names'][i])
    img = Image.open(image_path).convert('RGB')
    img = self.transform(img)
    return img 

  def __len__(self):
    return len(self.meta['image_names'])

#########################################################################
class TransformLoader:
  def __init__(self, image_size):
    self.image_size = image_size
    self.s = 1

  def get_composed_transform(self):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
    kernel_size = int(0.1 * self.image_size)
    kernel_size += (1-kernel_size%2)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.image_size, scale=(0.5, 1.0)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          GaussianBlur(kernel_size=kernel_size),
                                          transforms.ToTensor()])
    return data_transforms

class DataManager:
  @abstractmethod
  def get_data_loader(self, data_file):
    pass

class ContrastiveDataManager(DataManager):
  def __init__(self, image_size, batch_size):
    super(ContrastiveDataManager, self).__init__()
    self.batch_size = batch_size
    self.trans_loader = TransformLoader(image_size)

  def get_data_loader(self, data_file): #parameters that would change on train/val set
    transform = self.trans_loader.get_composed_transform()
    dataset = ContrastiveDataset(data_file, transform)
    data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = NUM_WORKERS, pin_memory = True)
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

    return data_loader



if __name__ == '__main__':
    base_datamgr            = SetDataManager(224, n_query = 16, n_support = 5)
    base_loader             = base_datamgr.get_data_loader(aug = True)

