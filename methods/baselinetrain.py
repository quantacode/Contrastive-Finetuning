from methods import backbone
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
import ipdb

# --- conventional supervised training ---
class BaselineTrain(nn.Module):
  def __init__(self, model_func, num_class, loadpath = None, tf_path=None, loss_type = 'softmax', flatten=True):
    super(BaselineTrain, self).__init__()
    self.method = 'baseline'
    
    # feature encoder
    self.feature    = model_func(flatten=flatten, leakyrelu=False)

    # loss function: use 'dist' to pre-train the encoder for matchingnet, and 'softmax' for others
    if loss_type == 'softmax':
      self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
      self.classifier.bias.data.fill_(0)
    elif loss_type == 'dist':
      self.classifier = backbone.distLinear(self.feature.final_feat_dim, num_class)
    self.loss_type = loss_type
    self.loss_fn = nn.CrossEntropyLoss()

    self.num_class = num_class
    self.tf_writer = SummaryWriter(log_dir=tf_path) if tf_path is not None else None

    if loadpath is not None:
      self.load_baseline(loadpath)

  def load_baseline(self, loadpath):
    state = torch.load(loadpath)['state']
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

  def train_loop(self, epoch, train_loader, optimizer, total_it):
    print_freq = len(train_loader) // 10
    avg_loss=0

    progress = tqdm(train_loader)
    for i, (x,y) in enumerate(progress):
      optimizer.zero_grad()
      loss = self.forward_loss(x, y)
      loss.backward()
      optimizer.step()

      avg_loss = avg_loss+loss.item()#data[0]

      progress.set_description('Epoch {:d} | Loss {:f}'.format(epoch, avg_loss/float(i+1)  ))
      if (total_it + 1) % 10 == 0:
        self.tf_writer.add_scalar('train/loss', loss.item(), total_it + 1)
      total_it += 1
    return total_it

###########################################################
  def test_loop(self, test_loader, num_samples=5):
    acc_all = []
    iter_num = len(test_loader)
    for i, (x, y) in enumerate(test_loader):
      scores = self.forward(x)
      acc = scores.cpu().argmax(axis=1)==y
      acc_all.extend(list(acc.numpy()))
      if i>num_samples:break
    acc_all = np.asarray(acc_all*100)
    acc_mean = np.mean(acc_all)
    print('--- %d Test Acc = %4.2f%% ---' % (i, acc_mean*100))
    return acc_mean

  def fewshot_task_loss(self, x, n_way, n_support, n_query):
    y_query = torch.from_numpy(np.repeat(range(n_way), n_query))
    y_query = y_query.cuda()
    x = x.contiguous().view(n_way * (n_support + n_query), *x.size()[2:])
    z_all_linearized = self.forward(x)
    z_all = z_all_linearized.view(n_way, n_support + n_query, -1)
    z_support = z_all[:, :n_support]
    z_query = z_all[:, n_support:]
    z_support = z_support.contiguous()
    z_proto = z_support.view(n_way, n_support, -1).mean(1)  # the shape of z is [n_data, n_dim]
    z_query = z_query.contiguous().view(n_way * n_query, -1)

    # normalize
    z_proto = F.normalize(z_proto, dim=1)
    z_query = F.normalize(z_query, dim=1)

    scores = cosine_dist(z_query, z_proto)
    loss = self.loss_fn(scores, y_query)
    return scores, loss, z_all_linearized

  def validate(self, n_way, n_support, n_query, x, epoch):
    self.eval()
    scores, loss, z_all = self.fewshot_task_loss(x, n_way, n_support, n_query)
    y_query = np.repeat(range(n_way), n_query)
    topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = np.sum(topk_ind[:, 0] == y_query)
    correct_this, count_this = float(top1_correct), len(y_query)
    acc_after = correct_this / count_this 
    self.tf_writer.add_scalar('validation/acc_before_training', acc_after, epoch + 1)
    return acc_after, loss, z_all

  def few_shot_validate(self, n_way, n_support, n_query, loader, epoch):
    acc_all = []
    for task_id, (x, y) in enumerate(loader):
      acc,_ ,_  = self.validate(n_way, n_support, n_query, x, epoch)
      acc_all.append(acc)
    return np.mean(acc_all)

def euclidean_dist( x, y):
  # x: N x D
  # y: M x D
  n = x.size(0)
  m = y.size(0)
  d = x.size(1)
  assert d == y.size(1)

  x = x.unsqueeze(1).expand(n, m, d)
  y = y.unsqueeze(0).expand(n, m, d)

  return torch.pow(x - y, 2).sum(2)


def cosine_dist(x, y):
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
