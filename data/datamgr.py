# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from data.dataset import SimpleDataset,NamedDataset, SetDataset, SetDataset_AugEpisode, MultiSetDataset, \
  EpisodicBatchSampler,EpisodicBatchSampler_Unsuperv, MultiEpisodicBatchSampler
from abc import abstractmethod
# from data.autoaugment import ImageNetPolicy

NUM_WORKERS=2
class TransformLoader:
  def __init__(self, image_size,
      normalize_param = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      jitter_param = dict(Brightness=0.4, Contrast=0.4, Color=0.4), rot_deg = 30,
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
      if self.is_aug_valoader:
        transform_list = ['RandomResizedCrop', 'AutoAugment', 'ToTensor', 'Normalize']
      elif self.no_color:
        transform_list = ['Grayscale', 'RandomResizedCrop', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
      else:
        transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
    else:
      if self.no_color:
        transform_list = ['Grayscale', 'Resize','CenterCrop', 'ToTensor', 'Normalize']
      elif self.only_resize:
        transform_list = ['Resize', 'ToTensor']
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

#class DataManager:
class DataManager(object):
  @abstractmethod
  def get_data_loader(self, data_file, aug):
    pass

class SimpleDataManager(DataManager):
  def __init__(self, image_size, batch_size, no_color=False, drop_last=False, is_shuffle=True):
    super(SimpleDataManager, self).__init__()
    self.batch_size = batch_size
    self.trans_loader = TransformLoader(image_size, no_color=no_color)
    self.drop_last = drop_last
    self.is_shuffle = is_shuffle

  def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
    transform = self.trans_loader.get_composed_transform(aug)
    dataset = SimpleDataset(data_file, transform)
    data_loader_params = dict(batch_size = self.batch_size, shuffle = self.is_shuffle,
                              num_workers = NUM_WORKERS, pin_memory =True, drop_last=self.drop_last)
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

    return data_loader

class NamedDataManager(SimpleDataManager):
  def __init__(self, image_size, batch_size, is_shuffle=True):
    super(NamedDataManager, self).__init__(image_size, batch_size)
    self.is_shuffle = is_shuffle

  def get_data_loader(self, data_file, aug):  # parameters that would change on train/val set
    transform = self.trans_loader.get_composed_transform(aug)
    dataset = NamedDataset(data_file, transform)
    data_loader_params = dict(batch_size=self.batch_size, shuffle=self.is_shuffle,
                              num_workers = NUM_WORKERS, pin_memory=True)
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
    return data_loader

class SetDataManager(DataManager):
  def __init__(self, image_size, n_way, n_support, n_query, n_episode=100, no_color=False):
    super(SetDataManager, self).__init__()
    self.image_size = image_size
    self.n_way = n_way
    self.n_support = n_support
    self.batch_size = n_support + n_query
    self.n_episode = n_episode

    self.trans_loader = TransformLoader(image_size, no_color=no_color)

  def get_data_loader(self, data_file, aug, ): #parameters that would change on train/val set
    transform = self.trans_loader.get_composed_transform(aug)
    if isinstance(data_file, list):
      dataset = MultiSetDataset( data_file , self.batch_size, transform )
      sampler = MultiEpisodicBatchSampler(dataset.lens(), self.n_way, self.n_episode )
    else:
      dataset = SetDataset( data_file , self.batch_size, transform )
      sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode )
    data_loader_params = dict(batch_sampler = sampler,  num_workers = NUM_WORKERS)
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
    return data_loader

class SetDataManager_AugEpisode(DataManager):
  def __init__(self, image_size, n_way, n_support, n_query, n_episode=100, workers=8):
    super(SetDataManager_AugEpisode, self).__init__()
    self.image_size = image_size
    self.n_way = n_way
    self.support_size = n_support
    self.batch_size = n_support + n_query
    self.n_episode = n_episode
    self.workers = workers

    self.trans_loader = TransformLoader(image_size, is_aug_valoader=True)

  def get_data_loader(self, data_file, aug=None):  # parameters that would change on train/val set
    tx = [self.trans_loader.get_composed_transform(aug=False), self.trans_loader.get_composed_transform(aug=True)]
    dataset = SetDataset_AugEpisode(data_file, self.batch_size, self.support_size, tx)
    sampler = EpisodicBatchSampler_Unsuperv(len(dataset), self.n_way, self.n_episode)
    data_loader_params = dict(batch_sampler = sampler,  num_workers=self.workers, pin_memory=True)
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
    return data_loader
