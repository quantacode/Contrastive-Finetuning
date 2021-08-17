import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from methods.protonet_cosine_based.mvcon import UFSL
from methods import backbone
import ipdb


class Transfer_Model(UFSL):
	def __init__(self, model_func, featype, tau, pos_wt_type, n_way, n_support,
	             tf_path=None, loadpath=None):
		super(Transfer_Model, self).__init__(model_func, tau, pos_wt_type, n_way, n_support, tf_path=tf_path,
		                                     loadpath=loadpath)
		self.method = 'Transfer_Model'
		self.featype = featype

	def fsl_loss(self, x, n_way, n_support, n_query):
		y_query = torch.from_numpy(np.repeat(range(n_way), n_query))
		y_query = y_query.cuda()
		x = x.cuda()
		x = x.contiguous().view(n_way * (n_support + n_query), *x.size()[2:])
		if self.featype == 'projection':
			z_all, _ = self.forward_this(x)
		elif self.featype == 'backbone':
			_, z_all = self.forward_this(x)
		# z_all = F.normalize(z_all, dim=1)
		z_all = z_all.view(n_way, n_support + n_query, -1)
		z_support = z_all[:, :n_support]
		z_query = z_all[:, n_support:]
		z_support = z_support.contiguous()
		z_proto = z_support.view(n_way, n_support, -1).mean(1)  # the shape of z is [n_data, n_dim]
		z_query = z_query.contiguous().view(n_way * n_query, -1)

		# scores = cosine_dist(z_query, z_proto)
		dists = euclidean_dist(z_query, z_proto)
		scores = -dists

		loss = self.loss_fn(scores, y_query)
		return loss

	def train_FSMT(self, epoch, src_loader, tgt_loader, optimizer, loss_wt = 1.0):
		# Baseline-source + MVcon-target
		print_freq = 10
		assert len(src_loader)==len(tgt_loader)
		src_loader = iter(src_loader)
		tgt_loader = iter(tgt_loader)
		for i in range(len(src_loader)):
			# source
			x_src, y_src = next(src_loader)
			n_query = x_src.size(1) - self.n_support
			n_way = x_src.size(0)
			loss_src = self.fsl_loss(x_src, n_way, self.n_support, n_query)
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
