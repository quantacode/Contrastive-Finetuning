import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from abc import abstractmethod
from tensorboardX import SummaryWriter
import ipdb

TRACKBN=True

class SIMCLR(nn.Module):
  def __init__(self, model_func, flatten=True, leakyrelu=False, tf_path=None, loadpath=None):
    super(SIMCLR, self).__init__()
    self.feature    = model_func(flatten=flatten, leakyrelu=leakyrelu)
    self.feat_dim   = self.feature.final_feat_dim
    self.fc = nn.Sequential(nn.Linear(self.feat_dim, 128), nn.BatchNorm1d(128, track_running_stats=TRACKBN))
    self.tf_writer = SummaryWriter(log_dir=tf_path) if tf_path is not None else None
    if loadpath != None:
      self.load_simclr(loadpath)
    if (torch.cuda.device_count() > 1):
      self.feature = nn.DataParallel(self.feature)
    self.method = 'SIMCLR'
    self.similarity = torch.nn.CosineSimilarity(dim=2)

  def load_simclr(self, loadpath):
    state = torch.load(loadpath)['state']
    self.load_state_dict(state, strict=False)
    return self

  def correct(self, x):
    scores, loss = self.set_forward_loss(x)
    y_query = np.repeat(range( self.n_way ), self.n_query )

    topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = np.sum(topk_ind[:,0] == y_query)
    return float(top1_correct), len(y_query), loss.item()*len(y_query)

  def contrastive_loss(self, x, x_tilde, temperature):
    bsz = x.shape[0]
    z_all = torch.cat([self.fc(self.feature(x)), self.fc(self.feature(x_tilde))], dim=0).unsqueeze(1)
    sim = self.similarity(z_all.repeat(1,2*bsz,1), z_all.transpose(0,1).repeat(2*bsz,1,1))
    S_all = torch.exp(sim/temperature)
    S_all = (S_all - S_all.diag().diag())
    V = torch.cat([S_all.diag(bsz), S_all.diag(-bsz)], dim=0)
    V = V/S_all.sum(dim=1)
    V = -torch.log(V)
    return V

  def train_contrastive(self, epoch, loader, optimizer, total_it, temperature):
    print_freq = len(loader) // 5
    avg_loss=0
    for i, (x1, x2) in enumerate(loader):
      optimizer.zero_grad()
      loss = self.contrastive_loss(x1.cuda(), x2.cuda(), temperature)
      loss = loss.mean()

      loss.backward()
      optimizer.step()
      avg_loss = avg_loss+loss.item()

      if (i + 1) % print_freq==0:
        print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i + 1, len(loader), avg_loss/float(i+1)))
      if (total_it + 1) % 10 == 0 and self.tf_writer is not None:
        self.tf_writer.add_scalar(self.method + '/total_loss', loss.item(), total_it + 1)
      total_it += 1
    return total_it

  def test_contrastive(self, loader, temperature):
    loss_total = 0
    count = 0
    for i, (x1, x2) in enumerate(loader):
      loss_batch = self.contrastive_loss(x1.cuda(), x2.cuda(), temperature)
      loss_total += loss_batch.sum().item()
      count += loss_batch.shape[0]
      if i>20:
        break
    loss = loss_total/count
    print('--- Loss = %.2f ---' % (loss))
    return loss