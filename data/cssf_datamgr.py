# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from data.dataset import SimpleDataset, SetDataset,EpisodicBatchSampler
from data.datamgr import TransformLoader
# from data.autoaugment import ImageNetPolicy
from abc import abstractmethod
import json
import numpy as np
import os
from PIL import Image
import ipdb
import matplotlib.pyplot as plt
import torch.nn.functional as F

# novel
NW_setDM = 0
NW_labCB = 0
# source
NW_simpleDM = 0
NW_unlabCB = 0
identity = lambda x:x

class DataManager:
  @abstractmethod
  def get_data_loader(self, data_file, aug):
    pass

##############################################################################
class SimpleDataManager(DataManager):
  def __init__(self, image_size, batch_size):
    super(SimpleDataManager, self).__init__()
    self.batch_size = batch_size
    self.trans_loader = TransformLoader(image_size, only_resize=True)

  def get_data_loader(self, data_file, shuffle=False):
    transform = self.trans_loader.get_composed_transform(aug=False)
    dataset = SimpleDataset(data_file, transform)
    data_loader_params = dict(batch_size = self.batch_size, shuffle = shuffle,
                              num_workers = NW_simpleDM, pin_memory = True,
                              drop_last=True)
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
    return data_loader

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

    self.trans_loader_ft = TransformLoader(image_size, only_resize=True)
    # self.trans_loader = TransformLoader(image_size, no_color=no_color)
    self.trans_loader = TransformLoader(image_size)

  def get_data_loader(self, data_file): #parameters that would change on train/val set
    transform_ft = self.trans_loader_ft.get_composed_transform( aug=False)
    transform_test = self.trans_loader.get_composed_transform( aug=False)
    dataset = SetDataset( data_file , self.batch_size, self.num_aug, transform_ft, transform_test)
    sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode )
    data_loader_params = dict(batch_sampler = sampler,  num_workers = NW_setDM)
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
    return data_loader

class SetDataset:
  def __init__(self, data_file, batch_size, num_aug, transform_ft, transform_test):
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
        pin_memory = False)
    for cl in self.cl_list:
      sub_dataset = SubDataset(self.sub_meta[cl], cl, num_aug=num_aug,
                               transform_ft = transform_ft, transform_test =transform_test )
      self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

  def __getitem__(self,i):
    return next(iter(self.sub_dataloader[i]))

  def __len__(self):
    return len(self.cl_list)

class SubDataset:
  def __init__(self, sub_meta, cl, num_aug=100,
               transform_ft=transforms.ToTensor(),transform_test=transforms.ToTensor(),
               target_transform=identity, min_size=50):
    self.sub_meta = sub_meta
    self.cl = cl
    # self.transform = ContrastiveWrapper(transform, num_aug)
    self.transform_ft = transform_ft
    self.transform_test = transform_test
    self.target_transform = target_transform
    if len(self.sub_meta) < min_size:
      idxs = [i % len(self.sub_meta) for i in range(min_size)]
      self.sub_meta = np.array(self.sub_meta)[idxs].tolist()

  def __getitem__(self,i):
    image_path = self.sub_meta[i]
    img_raw = Image.open(image_path).convert('RGB')
    img = self.transform_test(img_raw)
    img_ft = self.transform_ft(img_raw)
    target = self.target_transform(self.cl)
    return img, img_ft, target

  # def __getitem__(self,i):
  #   image_path = self.sub_meta[i]
  #   img_raw = Image.open(image_path).convert('RGB')
  #   # img_raw = Image.fromarray(plt.imread(image_path)).convert('RGB')
  #
  #   # if 'aircraft/images' in image_path:
  #   #   img = img.crop((0, 0, img.size[0], img.size[1] - 20))
  #
  #   # img_aug = self.transform(img)
  #   img = self.transform_test(img_raw)
  #   target = self.target_transform(self.cl)
  #   # return img, img_aug, target
  #   img_raw.close()
  #   return img, image_path, target

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
    self.x_pil = []
    if len(x.shape) > 2:
      self.x = x.contiguous().view(x.shape[0] * x.shape[1], *x.size()[2:])
      for bsz in range(self.x.size(0)):
        self.x_pil.append(transforms.ToPILImage()(self.x[bsz]))
    else:
      self.x = x.reshape(x.shape[0] * x.shape[1])
    self.transform = transform
    if y is not None:
      self.y = y.contiguous().view(-1)
    else:
      self.y = ['None']*self.x.shape[0]

  def __len__(self):
    return self.x.shape[0]

  def __getitem__(self, i):
    # img_raw = Image.open(self.x[i]).convert('RGB')
    # img = self.transform(img_raw)
    img = self.transform(self.x_pil[i])
    target = self.y[i]
    return img, target

class ContrastiveBatchifier:
  def __init__(self, n_way, n_support, image_size):
    self.n_way=n_way
    self.n_support=n_support
    self.transform = TransformLoader(image_size).get_composed_transform(aug=True)

  def get_loader(self, x):
    dataset = Augmentator(x, self.transform)
    data_loader_params = dict(
      dataset=dataset,
      batch_size = len(dataset),
      shuffle = False, # batchify fn below expects this to be False
      num_workers = NW_labCB,
      pin_memory = True)
    data_loader = torch.utils.data.DataLoader(**data_loader_params)
    return data_loader

  def batchify(self, x):
    x = x.view(self.n_way, self.n_support,*x.shape[1:])
    permuted_idx = torch.cat([torch.randperm(self.n_support).unsqueeze(0) for _ in range(self.n_way)], dim=0)
    shots_per_way = self.n_support if self.n_support % 2 == 0 else self.n_support - 1
    permuted_idx = permuted_idx[:, :shots_per_way]
    permuted_idx = permuted_idx.view(self.n_way, shots_per_way, 1, 1, 1).expand(
      self.n_way, shots_per_way, *x.shape[2:]).to(x.device)
    x = torch.gather(x, 1, permuted_idx)
    x = x.view(-1, *x.shape[2:])
    bch = torch.split(x, 2)
    bch = torch.cat([b.unsqueeze(0) for b in bch], dim=0)

    # for i in range(bch.shape[0]):
    #   for j in range(bch.shape[1]):
    #     img = bch[i, j].permute(1, 2, 0).cpu().numpy()
    #     img = (img - img.min()) / (img.max() - img.min())
    #     plt.imsave('%d%d.png' % (i, j), img)
    # ipdb.set_trace()
    return bch, shots_per_way

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

class UData_Batchifier:
  def __init__(self, batch_size, image_size):
    self.batch_size = batch_size
    self.transform = TransformLoader(image_size).get_composed_transform(aug=True)

  def get_loader(self, x):
    dataset = Augmentator(x, self.transform)
    data_loader_params = dict(
      dataset=dataset,
      batch_size = self.batch_size,
      shuffle = True,
      num_workers = NW_unlabCB,
      pin_memory = True)
    data_loader = torch.utils.data.DataLoader(**data_loader_params)
    return data_loader

class LData_Batchifier:
  def __init__(self, batch_size, image_size):
    self.batch_size = batch_size
    self.transform = TransformLoader(image_size).get_composed_transform(aug=True)

  def get_loader(self, x, y):
    dataset = Augmentator(x, self.transform, y)
    data_loader_params = dict(
      dataset=dataset,
      batch_size = self.batch_size,
      shuffle = True,
      num_workers = NW_unlabCB,
      pin_memory = True)
    data_loader = torch.utils.data.DataLoader(**data_loader_params)
    return data_loader


if __name__=="__main__":
  x = torch.rand(5,5,3,224,224)
  Contrastive_batchify(x)
