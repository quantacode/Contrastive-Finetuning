# only meant for DCL code
# BASIM : Baseline + simclr on the source
import os
import torch
import numpy as np
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from data.datamgr import TransformLoader as Baseline_TransformLoader
from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod
import json
from PIL import Image
from PIL import ImageFile
from data.gaussian_blur import GaussianBlur
ImageFile.LOAD_TRUNCATED_IMAGES = True
import ipdb
import sys
import numpy as np
sys.path.append("../")
from configs import *
import matplotlib.pyplot as plt
from data.autoaugment import ImageNetPolicy

NUM_WORKERS=8

class ContrastiveWrapper(object):
  def __init__(self, transform, num_aug=2):
    self.transform = transform
    self.num_aug = num_aug

  def __call__(self, sample):
    xlist = []
    for i in range(self.num_aug):
      xlist.append(self.transform(sample).unsqueeze(0))
    return torch.cat(xlist,dim=0)


class ContrastiveDataset:
  def __init__(self, data_file, transform):
    with open(data_file, 'r') as f:
      self.meta = json.load(f)
    self.labels = self.meta['image_labels']
    self.classes = np.unique(self.labels)
    self.transform = transform

  def __getitem__(self,i):
    image_path = os.path.join(self.meta['image_names'][i])
    # img_anchor = Image.fromarray(plt.imread(image_path)).convert('RGB')
    img_anchor = Image.open(image_path).convert('RGB')
    pos1 = self.transform(img_anchor)
    pos2 = self.transform(img_anchor)
    img = torch.cat([pos1.unsqueeze(0), pos2.unsqueeze(0)], dim=0)
    return img, self.meta['image_labels'][i]

  def __len__(self):
    return len(self.meta['image_names'])

class BasimDataset:
  def __init__(self, data_file, baseline_transform, transform):
    with open(data_file, 'r') as f:
      self.meta = json.load(f)
    self.labels = self.meta['image_labels']
    self.classes = np.unique(self.labels)
    self.transform = transform
    self.baseline_transform = baseline_transform

  def __getitem__(self,i):
    image_path = os.path.join(self.meta['image_names'][i])
    root_img = Image.open(image_path).convert('RGB')
    baseline_img = self.baseline_transform(root_img)
    pos1 = self.transform(root_img)
    pos2 = self.transform(root_img)
    img = torch.cat([pos1.unsqueeze(0), pos2.unsqueeze(0)], dim=0)
    return baseline_img, img, self.meta['image_labels'][i]

  def __len__(self):
    return len(self.meta['image_names'])


class SimpleDataset:
  def __init__(self, data_file, transform):
    with open(data_file, 'r') as f:
      self.meta = json.load(f)
    self.transform = transform

  def __getitem__(self,i):
    image_path = os.path.join(self.meta['image_names'][i])
    img = Image.open(image_path).convert('RGB')
    img = self.transform(img)
    target = self.meta['image_labels'][i]
    return img, target

  def __len__(self):
    return len(self.meta['image_names'])

#########################################################################
class TransformLoader:
  def __init__(self, image_size):
    self.image_size = image_size
    self.s = 1
    self.degrees = [0, 15, 30, 45]
    self.translation = [0.1, 0.25]
    self.pcnt = [0.05, 0.1, 0.2, 0.25, 0.5]

  def get_composed_transform(self, augmentation):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    if augmentation=='aggressive':
      color_jitter = dict(Brightness=0.9, Contrast=0.9, Color=0.9)
      data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.image_size, scale=(0.2, 1.0)),
                                            add_transforms.ImageJitter(color_jitter),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229,0.224, 0.225])])
    elif augmentation=='omniglot':
      degrees = self.degrees[0]
      translation = self.translation[1]
      pcnt = self.pcnt[1]
      data_transforms = transforms.Compose([transforms.RandomAffine(degrees=degrees,
                                                                    translate=(translation, translation)),
                                            transforms.ToTensor(),
                                            add_transforms.ZeroPixel(pcnt=pcnt),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229,0.224, 0.225])
                                            ])
    elif augmentation=='miniImagenet':
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.image_size, scale=(0.2, 1.0)),
                                              transforms.RandomHorizontalFlip(),
                                              ImageNetPolicy(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    elif augmentation=='celeba':
        color_jitter = transforms.ColorJitter(0.4 * self.s, 0.4 * self.s, 0.4 * self.s, 0.4 * self.s)
        kernel_size = int(0.1 * self.image_size)
        kernel_size += (1 - kernel_size % 2)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.image_size, scale=(0.2, 1.0)),
                                              transforms.RandomHorizontalFlip(p=0.5),
                                              # transforms.RandomRotation(30),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              # GaussianBlur(kernel_size=kernel_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    else:# augmentation == 'relaxed':
        color_jitter = transforms.ColorJitter(0.4 * self.s, 0.4 * self.s, 0.4 * self.s, 0.4 * self.s)
        kernel_size = int(0.1 * self.image_size)
        kernel_size += (1 - kernel_size % 2)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.image_size),
                                              transforms.RandomHorizontalFlip(p=0.5),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=kernel_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return data_transforms

class DataManager:
  @abstractmethod
  def get_data_loader(self, data_file):
    pass

class ContrastiveDataManager(DataManager):
  def __init__(self, image_size, batch_size, is_basim=False):
    super(ContrastiveDataManager, self).__init__()
    self.batch_size = batch_size
    self.trans_loader = TransformLoader(image_size)
    self.is_basim = is_basim
    if is_basim:
        self.baseline_tx = Baseline_TransformLoader(image_size)

  def get_data_loader(self, data_file, shuffle=False, augmentation='relaxed'): #parameters that would change on train/val set
    transform = self.trans_loader.get_composed_transform(augmentation)
    if self.is_basim:
        baseline_transform = self.baseline_tx.get_composed_transform(aug=True)
        dataset = BasimDataset(data_file, baseline_transform, transform)
    else:
        dataset = ContrastiveDataset(data_file, transform)

    data_loader_params = dict(batch_size = self.batch_size, shuffle = shuffle,
                              num_workers = NUM_WORKERS,
                              pin_memory = False, drop_last=True)
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
    return data_loader

class SimpleDataManager(DataManager):
  def __init__(self, image_size, batch_size):
    super(SimpleDataManager, self).__init__()
    self.batch_size = batch_size
    self.trans_loader = TransformLoader(image_size)

  def get_data_loader(self, data_file, shuffle=False, augmentation='relaxed'):
    transform = self.trans_loader.get_composed_transform(augmentation)
    dataset = SimpleDataset(data_file, transform)
    data_loader_params = dict(batch_size = self.batch_size, shuffle = shuffle,
                              num_workers = NUM_WORKERS, pin_memory = True,
                              drop_last=True)
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
    return data_loader



if __name__ == '__main__':
    base_datamgr            = SetDataManager(224, n_query = 16, n_support = 5)
    base_loader             = base_datamgr.get_data_loader(aug = True)

