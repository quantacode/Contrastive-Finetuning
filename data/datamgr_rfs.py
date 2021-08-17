# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from abc import abstractmethod
import json
import os
import ipdb
from PIL import Image
from PIL import ImageFile

NUM_WORKERS=2
MEAN = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
STD = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
class TransformLoader:
  def __init__(self, image_size,
      normalize_param = dict(mean=MEAN,std=STD),
      jitter_param = [0.4,0.4,0.4], rot_deg = 30,
      # jitter_param=dict(Brightness=0.9, Contrast=0.9, Color=0.9), rot_deg=45,
               is_aug_valoader = False, no_color=False, only_resize=False):
    self.image_size = image_size
    self.normalize_param = normalize_param
    self.jitter_param = jitter_param
    self.rot_deg = rot_deg
    self.is_aug_valoader = is_aug_valoader
    self.no_color = no_color
    self.only_resize = only_resize

  def parse_transform(self, transform_type):
    if transform_type=='ImageJitter':
      method = add_transforms.ImageJitter( self.jitter_param )
      return method
    method = getattr(transforms, transform_type)

    if transform_type=='Grayscale':
      return method(3)
    elif transform_type=='ColorJitter':
      return method(self.jitter_param[0],self.jitter_param[1],self.jitter_param[2])
    elif transform_type=='RandomCrop':
      return method(self.image_size, padding=8)
    elif transform_type=='CenterCrop':
      return method(self.image_size)
    elif transform_type=='Resize':
      # return method([int(self.image_size*1.15), int(self.image_size*1.15)])
      return method([self.image_size, self.image_size])
    elif transform_type=='RandomRotation':
      return method(self.rot_deg)
    elif transform_type=='Normalize':
      return method(**self.normalize_param )
    else:
      return method()

  def get_composed_transform(self, aug = False):
    if aug:
      transform_list = ['Resize', 'RandomCrop', 'ColorJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
      # transform_list = ['RandomCrop', 'ColorJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
    else:
      transform_list = ['Resize', 'ToTensor', 'Normalize']

    transform_funcs = []
    for x in transform_list:
      if x=='AutoAugment':
        transform_funcs.append(ImageNetPolicy())
      else:
        transform_funcs.append(self.parse_transform(x))
    transform = transforms.Compose(transform_funcs)
    return transform

#class DataManager:
class DataManager(object):
  @abstractmethod
  def get_data_loader(self, data_file, aug):
    pass

class SimpleDataset:
  def __init__(self, data_file, transform, target_transform=lambda x:x):
    with open(data_file, 'r') as f:
      self.meta = json.load(f)
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

class SimpleDataManager(DataManager):
  def __init__(self, image_size, batch_size, no_color=False, drop_last=False):
    super(SimpleDataManager, self).__init__()
    self.batch_size = batch_size
    self.trans_loader = TransformLoader(image_size, no_color=no_color)
    self.drop_last = drop_last

  def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
    transform = self.trans_loader.get_composed_transform(aug)
    dataset = SimpleDataset(data_file, transform)
    data_loader_params = dict(batch_size = self.batch_size, shuffle = True,
                              num_workers = NUM_WORKERS, pin_memory =True, drop_last=self.drop_last)
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

    return data_loader
