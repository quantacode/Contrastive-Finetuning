import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from methods.protonet_cosine import ProtoNet_cosine
from methods import backbone
import ipdb

class UFSL(ProtoNet_cosine):
  maml=False
  def __init__(self, model_func,  n_way, n_support, n_support_u=1, tf_path=None, loadpath=None, proj_head=False):
    super(UFSL, self).__init__(model_func,  n_way, n_support, tf_path=tf_path)

    if loadpath!=None:
      self.load_ufsl(loadpath)

    if (torch.cuda.device_count() > 1):
      self.distribute_model()

    if proj_head:
      self.projected_feature = nn.Sequential(
        self.feature,
        nn.Linear(self.feature.final_feat_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128)
      )
    else:
      self.projected_feature = self.feature

    self.method = 'UFSL'
    self.n_support_u = n_support_u

  def load_ufsl(self, loadpath):
    state = torch.load(loadpath)['state']
    self.load_state_dict(state, strict=False)
    return self

  def resume_best_ufsl(self, loadpath):
    loadfile = torch.load(loadpath)
    state = loadfile['state']
    self.load_state_dict(state, strict=False)
    resume_epoch = loadfile['epoch']
    return resume_epoch

  def set_forward_loss_target(self, x, feature_model):
    y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query_u))
    y_query = y_query.cuda()
    x = x.cuda()
    x = x.contiguous().view(self.n_way * (self.n_support_u + self.n_query_u), *x.size()[2:])
    z_all = feature_model.forward(x)
    z_all = z_all.view(self.n_way, self.n_support_u + self.n_query_u, -1)
    z_support = z_all[:, :self.n_support_u]
    z_query = z_all[:, self.n_support_u:]
    z_support = z_support.contiguous()
    z_proto = z_support.view(self.n_way, self.n_support_u, -1).mean(1)  # the shape of z is [n_data, n_dim]
    z_query = z_query.contiguous().view(self.n_way * self.n_query_u, -1)
    scores = cosine_dist(z_query, z_proto)
    loss = self.loss_fn(scores, y_query)
    return scores, loss

  def train_umtra(self, epoch, tgt_loader, optimizer, Nmb, batches_pb):
    avg_loss=0
    print_freq = 10
    assert print_freq%Nmb==0
    for i, (x_tgt,_) in enumerate(tgt_loader):
      self.n_query_u = x_tgt.size(1) - self.n_support_u
      self.n_way = x_tgt.size(0)
      _, loss = self.set_forward_loss_target(x_tgt, self.projected_feature)
      avg_loss = avg_loss+loss
      if i%Nmb==0:
        optimizer.zero_grad()
        avg_loss = avg_loss / float(Nmb)
        avg_loss.backward()
        optimizer.step()
        if i%print_freq==0:
          bch_id = epoch*batches_pb + int(i/Nmb)
          print('Batch {:d} | Loss {:f}'.format(bch_id, avg_loss.item()))
          self.tf_writer.add_scalar(self.method + '/query_loss/total', avg_loss.item(), bch_id)
        avg_loss = 0
    return

  def train_fsl_umtra(self, epoch, src_loader, tgt_loader, optimizer, loss_wt, Nmb, batches_pb):
    # trains src-protonet with tgt-umtra
    avg_loss_src = 0
    avg_loss_tgt = 0
    print_freq = 10
    src_loader = iter(src_loader)
    assert print_freq%Nmb==0
    assert len(src_loader)==len(tgt_loader)
    for i, (x_tgt, y_tgt) in enumerate(tgt_loader):
      x, y = next(src_loader)
      self.n_query = x.size(1) - self.n_support
      self.n_way  = x.size(0)
      _, loss_src = self.set_forward_loss(x)
      avg_loss_src += loss_src

      self.n_query_u = x_tgt.size(1) - self.n_support_u
      self.n_way = x_tgt.size(0)
      _, loss_tgt = self.set_forward_loss_target(x_tgt, self.projected_feature)
      avg_loss_tgt += loss_tgt

      optimizer.zero_grad()
      # update source loss every iteration
      if i % Nmb == 0:
        avg_loss_tgt /= float(Nmb)
        avg_loss = loss_src + loss_wt * avg_loss_tgt
        if i % print_freq == 0:
          bch_id = epoch*batches_pb + int(i/Nmb)
          avg_loss_src /= Nmb
          print('Batch {:d} | Loss_src {:f} | Loss_tgt {:f} | Loss {:f}'.format(
            bch_id, avg_loss_src.item(), avg_loss_tgt.item(), avg_loss.item()))
          self.tf_writer.add_scalar(self.method + '/query_loss/source', avg_loss_src.item(), bch_id)
          self.tf_writer.add_scalar(self.method + '/query_loss/target', avg_loss_tgt.item(), bch_id)
          self.tf_writer.add_scalar(self.method + '/query_loss/total', avg_loss.item(), bch_id)
        avg_loss_src=0
        avg_loss_tgt = 0
      else:
        avg_loss = loss_src
      avg_loss.backward()
      optimizer.step()
    return

  def sparsity_regularizer(self, x, model):
    z = model(x.cuda())
    bsz = z.shape[0]
    adjacency = cosine_dist(z, z)
    adjacency = torch.triu(adjacency, diagonal=1)
    loss = torch.abs(adjacency[adjacency!=0]).sum()/((bsz**2 - bsz)/2)
    return loss

  def train_fsl_L1(self, epoch, src_loader, tgt_loader, optimizer, loss_wt):
    # trains src-protonet with tgt-L1
    print_freq = 10
    assert len(src_loader) >= len(tgt_loader)
    for i in range(len(src_loader)):
      if i % len(src_loader) == 0:
        src_iter = iter(src_loader)
      x, y = next(src_iter)
      self.n_query = x.size(1) - self.n_support
      self.n_way  = x.size(0)
      _, loss_src = self.set_forward_loss(x)

      if i%len(tgt_loader)==0:
        tgt_iter = iter(tgt_loader)
      x_tgt, y_tgt = next(tgt_iter)
      loss_tgt = self.sparsity_regularizer(x_tgt, self.projected_feature)

      optimizer.zero_grad()
      # update source loss every iteration
      avg_loss = loss_src + loss_wt * loss_tgt
      avg_loss.backward()
      optimizer.step()
      if i % print_freq == 0:
        print('Epoch{:d} Iter {:d} | Loss_src {:f} | Loss_tgt {:f} | Loss {:f}'.format(
          epoch, i, loss_src.item(), loss_tgt.item(), avg_loss.item()))
        self.tf_writer.add_scalar(self.method + '/train/source', loss_src.item(), epoch*len(src_loader) + i)
        self.tf_writer.add_scalar(self.method + '/train/target-L1', loss_tgt.item(), epoch*len(src_loader) + i)
        self.tf_writer.add_scalar(self.method + '/train/total', avg_loss.item(), epoch*len(src_loader) + i)
    return

  def correct_target(self, x):
    scores, loss = self.set_forward_loss_target(x, self.feature)
    y_query = np.repeat(range(self.n_way), self.n_query_u)

    topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = np.sum(topk_ind[:, 0] == y_query)
    return float(top1_correct), len(y_query), loss.item() * len(y_query)

  def test_target(self, test_loader, record = None):
    loss = 0.
    count = 0
    acc_all = []

    iter_num = len(test_loader)
    for i, (x,y) in enumerate(test_loader):
      self.n_query_u = x.size(1) - self.n_support_u
      if self.change_way:
        self.n_way  = x.size(0)
      correct_this, count_this, loss_this = self.correct_target(x)
      acc_all.append(correct_this/ count_this*100  )
      loss += loss_this
      count += count_this

    acc_all  = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    print('--- %d Loss = %.6f ---' %(iter_num,  loss/count))
    print('--- %d Test Acc = %4.2f%% +- %4.2f%% ---' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
    return acc_mean


def cosine_dist( x, y):
  # x: N x D
  # y: M x D
  n = x.size(0)
  m = y.size(0)
  d = x.size(1)
  assert d == y.size(1)

  x = x.unsqueeze(1).expand(n, m, d)
  y = y.unsqueeze(0).expand(n, m, d)
  alignment = nn.functional.cosine_similarity(x, y, dim=2)
  return alignment