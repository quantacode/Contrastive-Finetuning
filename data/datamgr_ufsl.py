# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from data.dataset import SimpleDataset,NamedDataset, SetDataset, SetDataset_AugEpisode, MultiSetDataset, \
  EpisodicBatchSampler,EpisodicBatchSampler_Unsuperv, MultiEpisodicBatchSampler
from abc import abstractmethod
from data.autoaugment import ImageNetPolicy
NUM_WORKERS=2

class Miniimagenet_TransformLoader:
  def __init__(self, image_size):
    self.image_size = image_size
  def get_composed_transform(self, aug):
    # get a set of data aug transformations as described in the SimCLR paper.
    if aug:
      data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.image_size, scale=(0.2, 1.0)),
                                            transforms.RandomHorizontalFlip(),
                                            ImageNetPolicy(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    else:
      data_transforms = transforms.Compose([transforms.Resize([int(self.image_size*1.15), int(self.image_size*1.15)]),
                                            transforms.CenterCrop(self.image_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return data_transforms
  
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Omniglot_TransformLoader:
  def __init__(self, image_size, degrees=0, translation=0.25, pcnt=0.1):
    self.image_size = image_size
    self.s = 1
    self.degrees = degrees #[0, 30, 45]
    self.translation = translation #[0.1, 0.25]
    self.pcnt = pcnt #[0.1, 0.25, 0.5]
  def get_composed_transform(self, aug):
    # get a set of data aug transformations as described in the SimCLR paper.
    if aug:
      data_transforms = transforms.Compose([transforms.RandomAffine(degrees=self.degrees,
                                                                    translate=(self.translation, self.translation)),
                                            transforms.ToTensor(),
                                            add_transforms.ZeroPixel(pcnt=self.pcnt),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229,0.224, 0.225])])
    else:
      data_transforms = transforms.Compose([transforms.Resize([int(self.image_size*1.15), int(self.image_size*1.15)]),
                                            transforms.CenterCrop(self.image_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return data_transforms
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Celeba_TransformLoader:
  def __init__(self, image_size, degrees=0):
    self.image_size = image_size
    self.s = 1
  def get_composed_transform(self, aug):
    # get a set of data aug transformations as described in the SimCLR paper.
    if aug:
        color_jitter = transforms.ColorJitter(0.4 * self.s, 0.4 * self.s, 0.4 * self.s, 0.4 * self.s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.image_size, scale=(0.2, 1.0)),
                                              transforms.RandomHorizontalFlip(p=0.5),
                                              # transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              # GaussianBlur(kernel_size=kernel_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    else:
      data_transforms = transforms.Compose([transforms.Resize([int(self.image_size*1.15), int(self.image_size*1.15)]),
                                            transforms.CenterCrop(self.image_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return data_transforms
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



class DataManager:
  @abstractmethod
  def get_data_loader(self, data_file, aug):
    pass

class SimpleDataManager(DataManager):
  def __init__(self, image_size, batch_size, dataset):
    super(SimpleDataManager, self).__init__()
    self.batch_size = batch_size
    self.dataset = dataset
    if dataset=='omniglot':
      self.trans_loader = Omniglot_TransformLoader(image_size)
    elif dataset=='miniImagenet':
      self.trans_loader = Miniimagenet_TransformLoader(image_size)
    elif dataset=='celeba':
      self.trans_loader = Celeba_TransformLoader(image_size)

  def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
    transform = self.trans_loader.get_composed_transform(aug)
    dataset = SimpleDataset(data_file, transform)
    data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = NUM_WORKERS, pin_memory = True,
                              drop_last=True)
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
    return data_loader