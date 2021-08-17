from methods import backbone
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from methods.baselinetrain import BaselineTrain
import numpy as np
import ipdb

TRACKBN = True
# --- conventional supervised training ---
class BaCon(BaselineTrain):
  def __init__(self, model_func, num_class=64, tf_path=None, loadpath=None):
    super(BaCon, self).__init__(model_func, num_class, tf_path)
    self.feat_dim = self.feature.final_feat_dim
    self.fc = nn.Sequential(nn.Linear(self.feat_dim, 128), nn.BatchNorm1d(128, track_running_stats=TRACKBN))
    self.method = 'BaCon'
    self.similarity = torch.nn.CosineSimilarity(dim=2)
    if loadpath!=None:
      self.load_model(loadpath)
    if (torch.cuda.device_count() > 1):
      self.feature = nn.DataParallel(self.feature)

  def load_model(self, loadpath):
    state = torch.load(loadpath)['state']
    # newstate = {}
    # for key in state.keys():
    #   if 'classifier' not in key:
    #     newstate[key] = state[key]
    self.load_state_dict(state, strict=False)
    return self

  def forward(self,x):
    x = x.cuda()
    out  = self.feature.forward(x)
    scores  = self.classifier.forward(out)
    return scores

  def forward_loss(self, x, y):
    scores = self.forward(x)
    y = y.cuda()
    return self.loss_fn(scores, y )

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

  def train_combined(self, epoch, train_loader, target_loader, optimizer, total_it, temperature=0.5, contr_wt=1.0):
    print_freq = 10
    avg_loss=0

    for i, (x,y) in enumerate(train_loader):
      optimizer.zero_grad()
      # source
      loss_src = self.forward_loss(x, y)

      # target
      if i % (len(target_loader) - 1) == 0:
        tgl_iter = iter(target_loader)
      x_tgt1, x_tgt2 = next(tgl_iter)
      loss_tgt = self.contrastive_loss(x_tgt1.cuda(), x_tgt2.cuda(), temperature)
      loss_tgt = loss_tgt.mean()

      # combined
      loss = loss_src + contr_wt * loss_tgt

      loss.backward()
      optimizer.step()
      avg_loss = avg_loss+loss.item()#data[0]

      if (i + 1) % print_freq==0:
        print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i + 1, len(train_loader), avg_loss/float(i+1)  ))
      if (total_it + 1) % 10 == 0:
        self.tf_writer.add_scalar('train/loss/source', loss_src.item(), total_it + 1)
        self.tf_writer.add_scalar('train/loss/target', loss_tgt.item(), total_it + 1)
        self.tf_writer.add_scalar('train/loss/combined', loss.item(), total_it + 1)
      total_it += 1
    return total_it

  def test_contrastive(self, loader, temperature):
    loss_total = 0
    count = 0
    for i, (x1, x2) in enumerate(loader):
      loss_batch = self.contrastive_loss(x1.cuda(), x2.cuda(), temperature)
      loss_total += loss_batch.sum().item()
      count += loss_batch.shape[0]
    loss = loss_total/count
    print('--- Loss = %.2f ---' % (loss))
    return loss