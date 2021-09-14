# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
import json
import numpy as np
import os
from PIL import Image
import ipdb
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms

from abc import abstractmethod
from data.datamgr import TransformLoader


# novel
nWorker_setDM = 0
nWorker_labCB = 0
# source
nWorker_simpleDM = 0
nWorker_unlabCB = 0
identity = lambda x:x

class DataManager:
  @abstractmethod
  def get_data_loader(self, data_file, aug):
    pass

def list_collate(batch):
  xlist_batch = []
  y_batch = torch.Tensor([])
  for sample in batch:
    xlist_batch.append([sample[0]])
    y = sample[1]
    if isinstance(y, int):
      y = torch.Tensor([y])
    else:
      y = y.unsqueeze(0)
    y_batch = torch.cat([y_batch, y], dim=0)
  return xlist_batch, y_batch

def list_set_collate(batch):
  x_batch = torch.Tensor([])
  xlist_batch = []
  y_batch = torch.Tensor([])
  for sample in batch:
    x_batch = torch.cat([x_batch, sample[0].unsqueeze(0)], dim=0)
    xlist_batch.append(sample[1])
    y = sample[2]
    if isinstance(y, int):
      y = torch.Tensor([y])
    else:
      y = y.unsqueeze(0)
    y_batch = torch.cat([y_batch, y], dim=0)
  return x_batch, xlist_batch, y_batch

##############################################################################
class SimpleDataManager(DataManager):
  def __init__(self, batch_size):
    super(SimpleDataManager, self).__init__()
    self.batch_size = batch_size

  def get_data_loader(self, data_file, shuffle=False):
    dataset = SimpleDataset(data_file)
    data_loader_params = dict(batch_size = self.batch_size, shuffle = shuffle,
                              num_workers = nWorker_simpleDM, pin_memory = True,
                              drop_last=True, collate_fn=list_collate)
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
    return data_loader

class SimpleDataset:
  def __init__(self, data_file):
    with open(data_file, 'r') as f:
      self.meta = json.load(f)

  def __getitem__(self,i):
    image_path = os.path.join(self.meta['image_names'][i])
    img = Image.open(image_path).convert('RGB')
    target = self.meta['image_labels'][i]
    return img, target

  def __len__(self):
    return len(self.meta['image_names'])
##############################################################################

class SetDataManager(DataManager):
  def __init__(self, image_size, num_aug, n_way, n_support, n_query, n_episode=100, no_color=False):
    super(SetDataManager, self).__init__()
    self.image_size = image_size
    self.n_way = n_way
    self.n_support = n_support
    self.batch_size = n_support + n_query
    self.n_episode = n_episode
    self.num_aug = num_aug
    self.trans_loader = TransformLoader(image_size)

  def get_data_loader(self, data_file): #parameters that would change on train/val set
    transform_test = self.trans_loader.get_composed_transform( aug=False)
    dataset = SetDataset( data_file , self.batch_size, self.num_aug, transform_test)
    sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode )
    data_loader_params = dict(batch_sampler = sampler,  num_workers = nWorker_setDM, collate_fn=list_set_collate)
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
    return data_loader

class SetDataset:
  def __init__(self, data_file, batch_size, num_aug, transform_test):
    self.data_file = data_file
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
        pin_memory = False, collate_fn=list_set_collate)
    for cl in self.cl_list:
      sub_dataset = SubDataset(self.sub_meta[cl], cl, num_aug=num_aug,
                               transform_test =transform_test )
      self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

  def __getitem__(self,i):
    clswise_list = next(iter(self.sub_dataloader[i]))
    return clswise_list

  def __len__(self):
    return len(self.cl_list)

class SubDataset:
  def __init__(self, sub_meta, cl, num_aug=100, transform_test=transforms.ToTensor(),
               target_transform=identity, min_size=50):
    self.sub_meta = sub_meta
    self.cl = cl
    # self.transform = ContrastiveWrapper(transform, num_aug)
    self.transform_test = transform_test
    self.target_transform = target_transform
    if len(self.sub_meta) < min_size:
      idxs = [i % len(self.sub_meta) for i in range(min_size)]
      self.sub_meta = np.array(self.sub_meta)[idxs].tolist()

  def __getitem__(self,i):
    image_path = self.sub_meta[i]
    img_raw = Image.open(image_path).convert('RGB')
    img = self.transform_test(img_raw)
    target = self.target_transform(self.cl)
    return img, img_raw, target


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
    # img_raw = Image.open(self.x[i]).convert('RGB')
    # img = self.transform(img_raw)
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
  def __init__(self, n_way, n_support, image_size, augstrength='0'):
    self.n_way=n_way
    self.n_support=n_support
    jitter_param = dict(Brightness=0.4, Contrast=0.4, Color=0.4)
    self.transform = TransformLoader(image_size, jitter_param=jitter_param).get_composed_transform(aug=True)

  def get_loader(self, x):
    if self.n_support>1:
      dataset = Augmentator(x, self.transform)
    else:
      dataset = UnsupCon_Augmentator(x, self.transform)
    data_loader_params = dict(
      dataset=dataset,
      batch_size = len(dataset),
      shuffle = False,# batchify fn below expects this to be False
      num_workers = nWorker_labCB,
      pin_memory = True)
    data_loader = torch.utils.data.DataLoader(**data_loader_params)
    return data_loader

  def _batchify(self, x, n_way, n_support):
    # converts into contrastive batch
    x = x.contiguous().view(n_way, n_support,*x.shape[1:])
    permuted_idx = torch.cat([torch.randperm(n_support).unsqueeze(0) for _ in range(n_way)], dim=0)
    shots_per_way = n_support if n_support % 2 == 0 else n_support - 1
    permuted_idx = permuted_idx[:, :shots_per_way]
    permuted_idx = permuted_idx.view(n_way, shots_per_way, 1, 1, 1).expand(
      n_way, shots_per_way, *x.shape[2:]).to(x.device)
    x = torch.gather(x, 1, permuted_idx)
    x = x.view(-1, *x.shape[2:])
    bch = torch.split(x, 2)
    bch = torch.cat([b.unsqueeze(0) for b in bch], dim=0)
    return bch, shots_per_way

  def batchify(self, x):
    if self.n_support>1:
      return self._batchify(x, self.n_way, self.n_support)
    else:
      return x, 2 # gget_loader takes care of the form of the input

  def hpm_batchify(self, x):
    #hard positive mining
    featdim = x.shape[1]
    x = x.view(self.n_way, self.n_support, featdim)
    # x: [5,5,512]
    leftx = x.unsqueeze(2).expand(self.n_way, self.n_support, self.n_support, featdim)
    rightx = x.unsqueeze(1).expand(self.n_way, self.n_support, self.n_support, featdim)
    alignment = F.cosine_similarity(leftx, rightx, dim=3) #[5ways,5,5]
    farpos_idx = alignment.argmin(dim=2) #[5way,5]
    farpos_idx = farpos_idx.unsqueeze(2).unsqueeze(3).expand(self.n_way,self.n_support,1,featdim)
    farpos_sample = torch.gather(rightx, 2, farpos_idx)
    bch = torch.cat([x.view(-1, featdim).unsqueeze(1),farpos_sample.view(-1, featdim).unsqueeze(1)], dim=1)
    shots_per_way = self.n_support*2
    return bch, shots_per_way

if __name__=="__main__":
  x = torch.rand(5,5,3,224,224)
  Contrastive_batchify(x)
