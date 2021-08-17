import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from methods.protonet_cosine_based.mvcon import UFSL as MVCON
from methods import backbone
import ipdb


class Transfer_Model(MVCON):
	def __init__(self, model_func, projtype, tau, pos_wt_type, n_way, tf_path=None, loadpath=None, is_distribute=False):
		super(Transfer_Model, self).__init__(model_func=model_func,
		                                     tau=tau, pos_wt_type=pos_wt_type,
		                                     n_way=n_way,
		                                     # tf_path=tf_path, loadpath=loadpath)
		                                     tf_path=tf_path)
		if projtype=='separate':
			self.projection_head_supcon = nn.Sequential(
				nn.Linear(self.feature.final_feat_dim, self.projected_feature_dim)
			)
		elif projtype=='same':
			self.projection_head_supcon = self.projection_head

		self.method = 'Transfer_Model'
		if loadpath != None:
			self.load_model(loadpath)
		if is_distribute:
			self.distribute_model1()

	def load_model(self, loadpath):
		state = torch.load(loadpath)['state']
		loadstate = {}
		for key in state.keys():
			if '.module' in key:
				newkey = key.replace('.module', '')
				if 'projection_head' in key:
					newkey = newkey.replace('projection_head.', 'projection_head_supcon.')
			else:
				newkey = key
			loadstate[newkey] = state[key]
		self.load_state_dict(loadstate, strict=False)
		return self

	def forward_supcon_projection(self, x):
		r = self.feature(x)
		if self.projection_head_supcon is not None:
			z = self.projection_head_supcon(r)
		else:
			z = r
		return z, r

	def _supcon_loss(self, z, n_way, n_s):
		# normalize
		z = F.normalize(z, dim=1)
		bsz, featdim = z.size()
		z_square = z.view(bsz, 1, featdim).repeat(1, bsz, 1)
		# similarity of vectorized nway kshot task
		Sv = nn.CosineSimilarity(dim=2)(z_square, z_square.transpose(1, 0))
		Sv = torch.exp(Sv / self.tau)
		denominator = (Sv - Sv.diag().diag()).sum(dim=1)
		# create appropriate mask to compute the numerator and denominator
		mask = torch.FloatTensor(np.kron(np.eye(n_way), np.ones((n_s, n_s)))).to(Sv.device)
		if self.pos_wt_type == 'uniform':
			Sv_norm = Sv / denominator
			T1 = torch.log(Sv_norm)
			# self loss, s_ii is omitted
			T1 = T1 - T1.diag().diag()
			T1_pos = mask * T1
			li = (1.0 / (n_s - 1)) * T1_pos.sum(dim=1)
		elif self.pos_wt_type == 'min':
			S = mask * Sv + (1 - mask) * Sv.max()
			numerator = S.min(dim=1)[0]
			li = torch.log(numerator / denominator)
		loss = -self.tau * li.sum(dim=0)
		return loss

	def supcon_loss(self, x, mode='train'):
		n_way = x.size(0)
		n_s = x.size(1)
		x = x.cuda()
		x = x.contiguous().view(n_way * n_s, *x.size()[2:])
		z, r = self.forward_supcon_projection(x)
		if mode == 'train':
			return self._supcon_loss(z, n_way, n_s)
		elif mode == 'val':
			lp = self._supcon_loss(z, n_way, n_s)
			return lp

	def train_SSMT(self, epoch, src_loader, tgt_loader, optimizer, loss_wt):
		# Supcon-Source + Mvcon-Target
		print_freq = 10
		assert len(src_loader)==len(tgt_loader)
		src_loader = iter(src_loader)
		tgt_loader = iter(tgt_loader)
		for i in range(len(src_loader)):
			# source
			x_src, y_src = next(src_loader)
			loss_src = self.supcon_loss(x_src, 'train')
			# target
			x_tgt, y_tgt = next(tgt_loader)
			loss_tgt = self.mvcon_loss(x_tgt, 'train')
			# combined
			loss = loss_src + loss_wt * loss_tgt

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if i % print_freq == 0:
				bch_id = epoch * len(tgt_loader) + i
				print('Epoch {:d} | Batch {:d} | Loss {:f}'.format(epoch, i, loss.item()))
				self.tf_writer.add_scalar(self.method + '/source_loss', loss_src.item(), bch_id)
				self.tf_writer.add_scalar(self.method + '/target_loss', loss_tgt.item(), bch_id)
				self.tf_writer.add_scalar(self.method + '/TOTAL_loss', loss.item(), bch_id)
		return

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
