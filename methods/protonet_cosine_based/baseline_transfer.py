import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from methods.protonet_cosine_based.mvcon import UFSL
from methods import backbone
from tqdm import tqdm
from utils import *
import ipdb

class Transfer_Model(UFSL):
	def __init__(self, model_func, baseline_featype, num_classes, tau, pos_wt_type, n_way, n_support,
	             tf_path=None, loadpath=None, is_distribute=False):
		super(Transfer_Model, self).__init__(model_func, tau, pos_wt_type, n_way, n_support, tf_path=tf_path,
		                                     loadpath=loadpath)
		self.method = 'Transfer_Model'
		self.baseline_featype = baseline_featype
		if baseline_featype=='projection':
			self.classifier = nn.Linear(self.projected_feature_dim, num_classes)
		elif baseline_featype == 'backbone':
			self.classifier = nn.Linear(self.feature.final_feat_dim, num_classes)
		if is_distribute:
			self.distribute_model1()

	def baseline_loss(self, x, y):
		x, y = x.cuda(), y.cuda()
		z, r = self.forward_this(x)
		if self.baseline_featype == 'projection':
			scores = self.classifier(z)
		elif self.baseline_featype == 'backbone':
			scores = self.classifier(r)
		y = y.cuda()
		return self.loss_fn(scores, y)

	def train_BSMT(self, epoch, src_loader, tgt_loader, optimizer, loss_wt = 1.0):
		# Baseline-source + MVcon-target
		print_freq = 10
		train_iters = np.max([len(src_loader), len(tgt_loader)])
		progress = tqdm(range(train_iters))
		for i in progress:
			# source
			if i % (len(src_loader) - 1) == 0:
				src_loader = iter(src_loader)
			x_src, y_src = next(src_loader)
			loss_src = self.baseline_loss(x_src, y_src)
			# target
			if i % (len(tgt_loader) - 1) == 0:
				tgt_loader = iter(tgt_loader)
			x_tgt, y_tgt = next(tgt_loader)
			ipdb.set_trace()
			loss_tgt = self.mvcon_loss(x_tgt, 'train')
			# combined
			loss = loss_src + loss_wt * loss_tgt

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			progress.set_description('Epoch {:d} | Loss {:f}'.format(epoch, loss.item()))
			if i % print_freq == 0:
				bch_id = epoch * len(tgt_loader) + i
				self.tf_writer.add_scalar(self.method + '/source_loss', loss_src.item(), bch_id)
				self.tf_writer.add_scalar(self.method + '/target_loss', loss_tgt.item(), bch_id)
				self.tf_writer.add_scalar(self.method + '/TOTAL_loss', loss.item(), bch_id)
		return

	def train_basim(self, epoch, src_loader, optimizer, loss_wt = 1.0):
		# Baseline-source + Simclr-source
		print_freq = 10
		progress = tqdm(src_loader)
		for i, (x_src, x_src_simclr, y_src) in enumerate(progress):
			# source
			loss_src = self.baseline_loss(x_src, y_src)
			# self supervised auxiliary loss
			loss_tgt = self.mvcon_loss(x_src_simclr, 'train')
			# combined
			loss = loss_src + loss_wt * loss_tgt

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			progress.set_description('Epoch {:d} | Loss {:f}'.format(epoch, loss.item()))
			if i % print_freq == 0:
				bch_id = epoch * len(src_loader) + i
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
