# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from data.dataset import SimpleDataset,NamedDataset, SetDataset, SetDataset_AugEpisode, MultiSetDataset, \
  EpisodicBatchSampler,EpisodicBatchSampler_Unsuperv, MultiEpisodicBatchSampler
from abc import abstractmethod
# from data.autoaugment import ImageNetPolicy
NUM_WORKERS=4

class GaussianBlur(object):
  # Implements Gaussian blur as described in the SimCLR paper
  def __init__(self, kernel_size, min=0.1, max=2.0):
    self.min = min
    self.max = max
    # kernel size is set to be 10% of the image height/width
    self.kernel_size = kernel_size

  def __call__(self, sample):
    sample = np.array(sample)

    # blur the image with a 50% chance
    prob = np.random.random_sample()

    if prob < 0.5:
      sigma = (self.max - self.min) * np.random.random_sample() + self.min
      sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

    return sample

class TransformLoader:
  def __init__(self, image_size,
      normalize_param = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])):
    self.image_size = image_size
    self.normalize_param = normalize_param

  def get_composed_transform(self, aug = False):
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    kernel_size = int(0.1 * self.image_size)
    kernel_size += (1 - kernel_size % 2)
    if aug:
      data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.image_size),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            GaussianBlur(kernel_size=kernel_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.normalize_param)])
    else:
      data_transforms = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(self.normalize_param)])
    return data_transforms

class DataManager:
  @abstractmethod
  def get_data_loader(self, data_file, aug):
    pass

class SimpleDataset:
  def __init__(self, data_file, transform, target_transform=identity):
    with open(data_file, 'r') as f:
      self.meta = json.load(f)
    self.transform = transform
    self.target_transform = target_transform

  def __getitem__(self, i):
    image_path = os.path.join(self.meta['image_names'][i])
    img = Image.open(image_path).convert('RGB')
    img = self.transform(img)
    target = self.target_transform(self.meta['image_labels'][i])
    return img, target

  def __len__(self):
    return len(self.meta['image_names'])


class SimpleDataManager(DataManager):
  def __init__(self, image_size, batch_size, drop_last=True):
    super(SimpleDataManager, self).__init__()
    self.batch_size = batch_size
    self.trans_loader = TransformLoader(image_size)
    self.drop_last = drop_last

  def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
    transform = self.trans_loader.get_composed_transform(aug)
    dataset = SimpleDataset(data_file, transform)
    data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 4, pin_memory = True,
                              drop_last=self.drop_last)
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

    return data_loader
