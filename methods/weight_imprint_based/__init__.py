import numpy as np
import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from tensorboardX import SummaryWriter

class WeightImprint(nn.Module):
	def __init__(self, model_func, tf_path=None, loadpath=None, is_distribute=False, flatten=True, leakyrelu=False):
		super(WeightImprint, self).__init__()
		self.method = 'WeightImprint'
		self.model_func=model_func
		self.feature = model_func(flatten=flatten, leakyrelu=leakyrelu)
		self.feat_dim = self.feature.final_feat_dim
		self.tf_path = tf_path
		self.tf_writer = SummaryWriter(log_dir=self.tf_path)
		self.loss_fn = nn.CrossEntropyLoss() 
		if loadpath != None:
			self.load_model(loadpath)
		if is_distribute:
			self.distribute_model()

	def load_model(self, loadpath):
		state = torch.load(loadpath)
		if 'state' in state.keys():
			state = state['state']
			loadstate = {}
			for key in state.keys():
				if 'feature.module' in key:
					loadstate[key.replace('feature.module', 'feature')] = state[key]
				else:
					loadstate[key] = state[key]
			self.load_state_dict(loadstate, strict=False)
		return self

	def distribute_model(self):
		self.feature = nn.DataParallel(self.feature)
		return self

	def get_feature(self, x):
		return self.feature(x)

	def fewshot_task_loss(self, x, n_way, n_support, n_query):
		y_query = torch.from_numpy(np.repeat(range(n_way), n_query))
		y_query = y_query.cuda()
		x = x.contiguous().view(n_way * (n_support + n_query), *x.size()[2:])
		z_all_linearized = self.get_feature(x)
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
		acc_after = correct_this / count_this * 100
		self.tf_writer.add_scalar('validation/acc_before_training', acc_after, epoch + 1)
		return acc_after, loss, z_all

	def _get_epi_dist(self, n_way, n_samples, z_all):
		# cluster spread and separation
		all_affinities = cosine_dist(z_all, z_all)
		# cluster spread
		T1 = np.eye(n_way)
		T2 = np.ones((n_samples, n_samples))
		mask_pos = torch.FloatTensor(np.kron(T1, T2)).to(all_affinities.device)
		cluster_spread = (1-all_affinities)*mask_pos
		cluster_spread = cluster_spread.sum(dim=1).contiguous().view(n_way, n_samples)
		cluster_spread = cluster_spread.sum(dim=1)/2
		cluster_spread = cluster_spread.mean(dim=0)

		#cluster_sep
		mask_neg = 1-mask_pos
		cluster_sep = (1 - all_affinities) * mask_neg
		cluster_sep = cluster_sep.sum(dim=1).contiguous().view(n_way, n_samples)
		cluster_sep = cluster_sep.sum(dim=1).mean(dim=0)
		return cluster_spread.item(), cluster_sep.item()

	def get_episode_distances(self, n_way, n_support, n_query, z_all):
		z_all = z_all.detach().cpu()
		n_samples = n_support+n_query

		z_all = F.normalize(z_all, dim=1)
		z_all_reshaped = z_all.contiguous().view(n_way, n_samples, z_all.shape[-1])
		z_support = z_all_reshaped[:,:n_support].contiguous().view(-1, z_all.shape[-1])
		z_query = z_all_reshaped[:,n_support:].contiguous().view(-1, z_all.shape[-1])
		cspread_support, csep_support = self._get_epi_dist(n_way, n_support, z_support)
		cspread_query, csep_query = self._get_epi_dist(n_way, n_query, z_query)
		if n_support==1:
			cspread_support = np.random.rand(1)[0]*0.00001

		return cspread_support, csep_support, cspread_query, csep_query


#########################################################
class LinearEvaluator(nn.Module):
	def __init__(self, feature, outdim, train_size):
		super(LinearEvaluator, self).__init__()
		self.feature = feature
		self.L = weight_norm(nn.Linear(feature.feat_dim, outdim, bias=False), name='weight', dim=0)
		self.loss_fn = nn.CrossEntropyLoss()
		self.batch_size = 64
		self.train_size = train_size
		self.train_iters = 50

	def get_scores(self, x):
		x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
		x_normalized = x.div(x_norm + 0.00001)
		L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
		self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
		cos_dist = self.L(x_normalized)
		scores = 10 * cos_dist
		return scores

	def train_classifier(self, input, target):
		classifier_opt = torch.optim.SGD(self.L.parameters(),
		                                 lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
		self.cuda()
		self.feature.eval()
		self.L.train()
		for epoch in range(self.train_iters):
			rand_id = np.random.permutation(self.train_size)
			for j in range(0, self.train_size, self.batch_size):
				classifier_opt.zero_grad()
				selected_id = torch.from_numpy(rand_id[j: min(j + self.batch_size, self.train_size)]).cuda()
				x_batch = input[selected_id]
				y_batch = target[selected_id]
				output = self.get_scores(self.feature.get_feature(x_batch))
				loss = self.loss_fn(output, y_batch)
				loss.backward()
				classifier_opt.step()

	def test_classifier(self, input, target):
		self.feature.eval()
		self.L.eval()
		scores = self.get_scores(self.feature.get_feature(input))
		topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
		topk_ind = topk_labels.cpu().numpy()
		top1_correct = np.sum(topk_ind[:, 0] == target)
		correct_this, count_this = float(top1_correct), len(target)
		acc = correct_this / count_this * 100
		return acc

	def evaluate(self, xtrain, ytrain, xtest, ytest):
		self.train_classifier(xtrain, ytrain)
		task_acc = self.test_classifier(xtest, ytest)
		return task_acc



#########################################################
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
