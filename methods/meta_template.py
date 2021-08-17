import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from abc import abstractmethod
from tensorboardX import SummaryWriter
from model_resnet import *
import ipdb

REP_FREQ = 100000
SIM_FREQ = 1010
class MetaTemplate(nn.Module):
  def __init__(self, model_func, n_way, n_support, flatten=True, leakyrelu=False, tf_path=None, change_way=True):
    super(MetaTemplate, self).__init__()
    self.n_way      = n_way
    self.n_support  = n_support
    self.n_query    = -1 #(change depends on input)
    if isinstance(model_func,str) and model_func == 'resnet18':
      self.feature = ResidualNet('ImageNet', 18, 1000, None, tracking=False, use_bn=True)
      self.feature.final_feat_dim = 512
      self.feat_dim = self.feature.final_feat_dim
    elif model_func.__name__ == 'resnet18_cnaps':
      self.feature = model_func()
      self.feat_dim = 512
    elif model_func.__name__ == 'resnet12':
      self.feature    = model_func(flatten=flatten, leakyrelu=leakyrelu)
      self.feat_dim = 640
    elif model_func.__name__ == 'resnet12_scl':
      self.feature    = model_func(flatten=flatten, leakyrelu=leakyrelu)
      self.feat_dim = self.feature.feat_dim
    else:
      self.feature    = model_func(flatten=flatten, leakyrelu=leakyrelu)
      self.feat_dim   = self.feature.final_feat_dim

    # if (torch.cuda.device_count() > 1):
    #   print("Let's use", torch.cuda.device_count(), "GPUs!")
    #   self.feature = nn.DataParallel(self.feature)
    self.change_way = change_way  #some methods allow different_way classification during training and test
    self.tf_writer = SummaryWriter(log_dir=tf_path) if tf_path is not None else None

  @abstractmethod
  def set_forward(self,x,is_feature):
    pass

  @abstractmethod
  def set_forward_loss(self, x):
    pass

  @abstractmethod
  def clusterReg(self, x):
    pass

  def forward(self,x):
    out  = self.feature.forward(x)
    return out

  def parse_feature(self,x,is_feature):
    x = x.cuda()
    if is_feature:
      z_all = x
    else:
      x           = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:])
      z_all       = self.feature.forward(x)
      z_all       = z_all.view( self.n_way, self.n_support + self.n_query, -1)
    z_support   = z_all[:, :self.n_support]
    z_query     = z_all[:, self.n_support:]
    return z_support, z_query

  def train_loop(self, epoch, train_loader, optimizer):
    print_freq = 10
    avg_loss=0
    for i, (x,_ ) in enumerate(train_loader):
      self.n_query = x.size(1) - self.n_support
      self.n_way = x.size(0)
      _, loss = self.set_forward_loss(x)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      avg_loss += loss.item()
      if i % print_freq == 0:
        # print(optimizer.state_dict()['param_groups'][0]['lr'])
        print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss / float(i + 1)))
    return

  def train_loop_contrastive(self, epoch, train_loader, target_loader, optimizer, total_it, temperature, contr_wt=1.0):
    print_freq = len(train_loader) // 10
    avg_loss=0
    for i, (x,_ ) in enumerate(train_loader):
      if i % (len(target_loader)-1) == 0:
        tgtloader_iter = iter(target_loader)
      x_tgt1, x_tgt2 = next(tgtloader_iter)

      self.n_query = x.size(1) - self.n_support
      if self.change_way:
        self.n_way  = x.size(0)
      optimizer.zero_grad()

      _, loss_fsl = self.set_forward_loss(x.cuda())
      loss_contrastive = self.set_contrastive_loss(x_tgt1.cuda(), x_tgt2.cuda(), temperature)

      loss = loss_fsl + contr_wt * loss_contrastive
      loss.backward()
      optimizer.step()
      avg_loss = avg_loss+loss.item()

      if (i + 1) % print_freq==0:
        print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i + 1, len(train_loader), avg_loss/float(i+1)))
      if (total_it + 1) % 10 == 0 and self.tf_writer is not None:
        self.tf_writer.add_scalar(self.method + '/query_loss', loss_fsl.item(), total_it + 1)
        self.tf_writer.add_scalar(self.method + '/contrast_loss', loss_contrastive.item(), total_it + 1)
        self.tf_writer.add_scalar(self.method + '/total_loss', loss.item(), total_it + 1)
      total_it += 1
    return total_it

  def train_simada(self, epoch, train_loader, target_loader, optimizer, total_it, cr_wt=1.0):

    def copy_params():
      params1 = self.named_parameters()
      dict_params2 = {}
      for name1, param1 in params1:
        if name1 not in dict_params2:
          dict_params2[name1] = torch.rand(param1.shape)
        dict_params2[name1].data.copy_(param1.data)
      return dict_params2

    def print_param_change(before, after):
      for key in before.keys():
        delta = ((before[key] != after[key]).sum() > 0).float()
        print("%s : %f" % (key, delta))
      return

    optim_fsl, optim_rep, optim_sim = optimizer
    print_freq = len(train_loader) // 10
    avg_loss = 0
    rep_update = True
    for i, (x,_ ) in enumerate(train_loader):
      self.n_query = x.size(1) - self.n_support
      if self.change_way: self.n_way  = x.size(0)
      optim_fsl.zero_grad()
      _, loss_fsl = self.set_forward_loss(x)
      loss_fsl.backward()
      optim_fsl.step()
      avg_loss = avg_loss + loss_fsl.item()

      # target losses
      if i % (len(target_loader) - 1) == 0:
        tgtloader_iter = iter(target_loader)
      x_tgt,_ = next(tgtloader_iter)

      if rep_update:
        optim_rep.zero_grad()
        reg = self.clusterReg(x_tgt.cuda())
        reg.backward()
        optim_rep.step()
      else:
        optim_sim.zero_grad()
        reg = self.clusterReg(x_tgt.cuda())
        reg.backward()
        optim_sim.step()

      counter = epoch*len(train_loader) + i
      if counter%SIM_FREQ==0 and counter>0:
        print('sim update!')
        rep_update = False
      else:
        rep_update = True

      if (i + 1) % print_freq==0:
        print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i + 1, len(train_loader), avg_loss/float(i+1)))
      if (total_it + 1) % 10 == 0 and self.tf_writer is not None:
        self.tf_writer.add_scalar(self.method + '/query_loss', loss_fsl.item(), total_it + 1)
        self.tf_writer.add_scalar(self.method + '/cluster_regularization', reg.item(), total_it + 1)
      total_it += 1
    return total_it


  def correct(self, x):
    scores, loss = self.set_forward_loss(x)
    y_query = np.repeat(range( self.n_way ), self.n_query )

    topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = np.sum(topk_ind[:,0] == y_query)
    return float(top1_correct), len(y_query), loss.item()*len(y_query)

  def test_loop(self, test_loader, record = None):
    loss = 0.
    count = 0
    acc_all = []

    iter_num = len(test_loader)
    for i, (x,y) in enumerate(test_loader):
      self.n_query = x.size(1) - self.n_support
      if self.change_way:
        self.n_way  = x.size(0)
      correct_this, count_this, loss_this = self.correct(x)
      acc_all.append(correct_this/ count_this*100  )
      loss += loss_this
      count += count_this

    acc_all  = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = 1.96*np.std(acc_all)/np.sqrt(iter_num)
    print('--- %d Loss = %.6f ---' %(iter_num,  loss/count))
    print('--- %d Test Acc = %4.2f%% +- %4.2f%% ---' %(iter_num,  acc_mean, acc_std))
    return acc_mean, acc_std

  def test_nmi(self, test_loader, record = None):
    raise NotImplementedError
