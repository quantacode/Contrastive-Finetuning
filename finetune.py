'''

'''
import numpy as np
import time
import os
import glob
from itertools import combinations
from tqdm import tqdm
import matplotlib.pyplot as plt
from shutil import copy
import pickle
import ipdb

import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable

from methods.backbone import model_dict
from methods.weight_imprint_based import LinearEvaluator
from methods.weight_imprint_based.ConCE import ConCeModel
from data import get_datafiles
from data.datamgr import SimpleDataManager
from data.cssf_datamgr_custom_collate import SetDataManager, ContrastiveBatchifier

import configs
from options import parse_args, get_best_file, get_assigned_file
from utils import *
from read_parallel_results import consolidate_results

EPS = 0
eps = 0.0000001

def record_each(q_type, quantity, results_path):
	for key in quantity.keys():
		# for each epoch
		if len(quantity[key]) != 0:
			quantity_this = quantity[key]
			quantity_this = np.asarray(quantity_this)
			with open(results_path.replace('results.txt', '%s%s.pkl' % (q_type, key)), 'wb') as f:
				pickle.dump(quantity_this, f)


def record(acc_all_le, acc_all, accDiff_all, cluster_support, cluster_query, results_path):
	acc_all_le = np.asarray(acc_all_le)
	
	with open(results_path.replace('results.txt', 'linev.pkl'), 'wb') as f:
		pickle.dump(acc_all_le, f)
	
	# accuracy
	record_each('wi_final', acc_all, results_path)
	record_each('wi_delta', accDiff_all, results_path)
	# support clusters
	record_each('wi_support_cspread', cluster_support['cspread'], results_path)
	record_each('wi_support_cspread_pcnt', cluster_support['cspread_pcnt'], results_path)
	record_each('wi_support_csep', cluster_support['csep'], results_path)
	record_each('wi_support_csep_pcnt', cluster_support['csep_pcnt'], results_path)
	# query clusters
	record_each('wi_query_cspread', cluster_query['cspread'], results_path)
	record_each('wi_query_cspread_pcnt', cluster_query['cspread_pcnt'], results_path)
	record_each('wi_query_csep', cluster_query['csep'], results_path)
	record_each('wi_query_csep_pcnt', cluster_query['csep_pcnt'], results_path)


def epoch_wise_collection(model, n_way, n_support, n_query, x, epoch,
													acc_before,
													cspread_support_before, csep_support_before, cspread_query_before, csep_query_before,
													acc_all, accDiff_all, cluster_support, cluster_query):
	# accuracies
	acc_after, _, z_all_analysis = model.validate(n_way, n_support, n_query, x, epoch)
	acc_all[str(epoch + 1)].append(acc_after)
	improvement = acc_after - acc_before
	accDiff_all[str(epoch + 1)].append(improvement)
	
	# cluster distances
	cspread_support_after, csep_support_after, cspread_query_after, csep_query_after = \
		model.get_episode_distances(n_way, n_support, n_query, z_all_analysis)
	# support spread
	delta = cspread_support_after - cspread_support_before
	cluster_support['cspread'][str(epoch + 1)].append(delta)
	cluster_support['cspread_pcnt'][str(epoch + 1)].append(delta / (cspread_support_before + eps))
	# support separation
	delta = csep_support_after - csep_support_before
	cluster_support['csep'][str(epoch + 1)].append(delta)
	cluster_support['csep_pcnt'][str(epoch + 1)].append(delta / (csep_support_before + eps))
	# query spread
	delta = cspread_query_after - cspread_query_before
	cluster_query['cspread'][str(epoch + 1)].append(delta)
	cluster_query['cspread_pcnt'][str(epoch + 1)].append(delta / (cspread_query_before + eps))
	# query separation
	delta = csep_query_after - csep_query_before
	cluster_query['csep'][str(epoch + 1)].append(delta)
	cluster_query['csep_pcnt'][str(epoch + 1)].append(delta / (csep_query_before + eps))
	# return acc_after, improvement, z_all_analysis, cspread_support_before, csep_support_before, cspread_query_before, \
	#        csep_query_before, acc_all, accDiff_all, cluster_support, cluster_query
	return acc_after, improvement, z_all_analysis, cspread_support_after, csep_support_after, cspread_query_after, \
				 csep_query_after, acc_all, accDiff_all, cluster_support, cluster_query


def get_init_var():
	acc_all_le = []
	acc_all = {'0': [], '10': [], '25': [], '50': [], '75': [], '100': [], '200': [], '300': [], '400': [], '500': [],
						 '600': [], '700': [], '800': [], '900': [], '1000': []}
	accDiff_all = {'0': [], '10': [], '25': [], '50': [], '75': [], '100': [], '200': [], '300': [], '400': [], '500': [],
								 '600': [], '700': [], '800': [], '900': [], '1000': []}
	cluster_support = {}
	cluster_support['cspread'] = {'0': [], '10': [], '25': [], '50': [], '75': [], '100': [], '200': [], '300': [],
																'400': [], '500': [], '600': [], '700': [], '800': [], '900': [], '1000': []}
	cluster_support['cspread_pcnt'] = {'0': [], '10': [], '25': [], '50': [], '75': [], '100': [], '200': [], '300': [],
																		 '400': [], '500': [], '600': [], '700': [], '800': [], '900': [], '1000': []}
	cluster_support['csep'] = {'0': [], '10': [], '25': [], '50': [], '75': [], '100': [], '200': [], '300': [],
														 '400': [], '500': [], '600': [], '700': [], '800': [], '900': [], '1000': []}
	cluster_support['csep_pcnt'] = {'0': [], '10': [], '25': [], '50': [], '75': [], '100': [], '200': [], '300': [],
																	'400': [], '500': [], '600': [], '700': [], '800': [], '900': [], '1000': []}
	cluster_query = {}
	cluster_query['cspread'] = {'0': [], '10': [], '25': [], '50': [], '75': [], '100': [], '200': [], '300': [],
															'400': [], '500': [], '600': [], '700': [], '800': [], '900': [], '1000': []}
	cluster_query['cspread_pcnt'] = {'0': [], '10': [], '25': [], '50': [], '75': [], '100': [], '200': [], '300': [],
																	 '400': [], '500': [], '600': [], '700': [], '800': [], '900': [], '1000': []}
	cluster_query['csep'] = {'0': [], '10': [], '25': [], '50': [], '75': [], '100': [], '200': [], '300': [], '400': [],
													 '500': [], '600': [], '700': [], '800': [], '900': [], '1000': []}
	cluster_query['csep_pcnt'] = {'0': [], '10': [], '25': [], '50': [], '75': [], '100': [], '200': [], '300': [],
																'400': [], '500': [], '600': [], '700': [], '800': [], '900': [], '1000': []}
	return acc_all_le, acc_all, accDiff_all, cluster_support, cluster_query


def finetune(source_loader, novel_loader, total_epoch, model_params, dataloader_params, params):
	results_dir = params.results_dir
	n_way, n_support = dataloader_params['n_way'], dataloader_params['n_support']
	image_size = dataloader_params['image_size']
	acc_all_le, acc_all, accDiff_all, cluster_support, cluster_query = get_init_var()
	
	# model
	model = ConCeModel(n_way=n_way, projhead=params.projection_head,
										 ft_mode=params.ft_mode, **model_params)
	progress = tqdm(novel_loader)
	for task_id, (x, x_ft, y) in enumerate(progress):
		if params.ufsl_dataset:
			supcon_datamgr = ContrastiveBatchifier(n_way=n_way, n_support=n_support, image_size=image_size,
																						 dataset=params.targetset)
		else:
			supcon_datamgr = ContrastiveBatchifier(n_way=n_way, n_support=n_support, image_size=image_size,
																						 augstrength=params.augstrength)
		
		if params.is_tgt_aug:
			supcon_dataloader = supcon_datamgr.get_loader([sample[:n_support] for sample in x_ft])
		else:
			supcon_dataloader = None
		###############################################################################################
		# load pretrained feature extractor
		model.refresh_from_chkpt()
		model.cuda()
		optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
		n_query = x.size(1) - n_support
		x = x.cuda()
		
		acc_before, _, z_all_analysis, cspread_support_before, csep_support_before, cspread_query_before, \
		csep_query_before, acc_all, accDiff_all, cluster_support, cluster_query = epoch_wise_collection(
			model, n_way, n_support, n_query, x, -1,
			EPS, EPS, EPS, EPS, EPS, acc_all, accDiff_all, cluster_support, cluster_query)
		
		for epoch in range(total_epoch):
			model.train()
			
			# labelled contrastive batch
			if supcon_dataloader is not None:
				x_l_fewshot, _ = next(iter(supcon_dataloader))
				x_l_fewshot = x_l_fewshot.cuda()
			else:
				# without any augmentation
				x_l_fewshot = x[:, :n_support, :, :, :]
				x_l_fewshot = x_l_fewshot.contiguous().view(x_l_fewshot.size(0) * x_l_fewshot.size(1),
																										*x_l_fewshot.shape[2:]).cuda()
			
			# unlabelled batch
			if epoch % len(source_loader) == 0:
				src_iter = iter(source_loader)
			x_src, y_src = next(src_iter)
			x_src, y_src = x_src.cuda(), y_src.cuda()
			unlab_bsz = x_src.size(0)
			z_src = model.get_feature(x_src)
			
			# Batchify
			x_l, shots_per_way = supcon_datamgr.batchify(x_l_fewshot)
			x_l = x_l.view(x_l.size(0) * x_l.size(1), *x_l.size()[2:])
			z_l = model.forward_this(x_l)
			# random paired positives
			z_u = model.forward_projection(z_src)
			z_batch = torch.cat([z_l, z_u], dim=0)
			loss_primary = model.cssf_loss(z_batch, shots_per_way, n_way, unlab_bsz,
																		 mode=params.ft_mode, alpha=params.alpha)
			
			if 'mtce' in params.ft_mode:
				loss_mt = model.CE_loss_source(z_src, y_src)
				loss = loss_primary + loss_mt
			else:
				# no Multitask supervised loss
				loss = loss_primary
			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			model.tf_writer.add_scalar('train/loss', loss.item(), epoch)
			if str(epoch + 1) in acc_all.keys():
				acc_after, improvement, _, _, _, _, _, acc_all, accDiff_all, cluster_support, cluster_query = epoch_wise_collection(
					model, n_way, n_support, n_query, x, epoch,
					acc_before,
					cspread_support_before, csep_support_before, cspread_query_before,
					csep_query_before,
					acc_all, accDiff_all, cluster_support, cluster_query)
				
				#progress.set_description('improvement%d = %0.3f, finalacc%d = %0.3f' % (epoch + 1, improvement, epoch + 1, acc_after))
				progress.set_description('epoch %d' % (epoch + 1))
		
		acc_all_le.append(0)
		if (task_id + 1) % 10 == 0 or task_id == len(novel_loader) - 1:
			record(acc_all_le, acc_all, accDiff_all, cluster_support, cluster_query,
						 results_path=os.path.join(results_dir, 'results.txt'))
	
if __name__ == '__main__':
	np.random.seed(10)
	params = parse_args('train/ufsl/cssf/parallel')
	print(params)
	resultdir = '%s/%s' % (params.save_dir, params.name)
	params.name = os.path.join(params.name, str(params.run_id))
	
	# output and tensorboard dir
	params.tf_dir = '%s/%s/log' % (params.save_dir, params.name)
	params.checkpoint_dir = params.tf_dir.replace('/log', '/chkpt')
	if not os.path.isdir(params.checkpoint_dir):
		os.makedirs(params.checkpoint_dir)
	params.results_dir = params.tf_dir.replace('/log', '/results')
	if not os.path.isdir(params.results_dir):
		os.makedirs(params.results_dir)
	##################################################################
	if 'Conv' in params.model:
		image_size = 84
	else:
		image_size = 224
	num_tasks = params.num_tasks
	total_epoch = params.stop_epoch
	
	# dataloaders
	target_file, target_val_file, target_novel_file = get_datafiles(params, configs)
	if params.hyperparam_select:
		inference_file = target_val_file
	else:
		inference_file = target_novel_file
	
	dataloader_params = dict(
		image_size=image_size,
		num_aug=total_epoch,
		n_way=params.test_n_way,
		n_support=params.n_shot,
		n_episode=num_tasks,
		n_query=params.n_query)
	
	if target_file is not None:
		novel_loader = SetDataManager(**dataloader_params).get_data_loader(inference_file)
		source_loader = SimpleDataManager(image_size, params.distractor_bsz).get_data_loader(
			target_file, aug=params.is_src_aug)
	else:
		raise ValueError('must define "--distractor_set" flag')
	
	print('\n------------ validating : %s --------------' % (params.hyperparam_select))
	print('finetuning mode : %s' % params.ft_mode)
	print('model : %s' % params.model)
	print('source file: %s' % (target_file))
	print('novel file: %s' % (inference_file))
	print('n_way: %d, n_shot: %d, n_query: %d' % (
		dataloader_params['n_way'], dataloader_params['n_support'], dataloader_params['n_query']))
	print('distractor sz: %s' % (params.distractor_bsz))
	print('tau : %s' % (params.tau))
	print('lr : %s' % (params.lr))
	print('Source Aug : ', params.is_src_aug)
	print('Target Aug : ', params.is_tgt_aug)
	
	# model
	model_params = dict(
		model_func=model_dict[params.model],
		tau=params.tau,
		tf_path=params.tf_dir,
		loadpath=params.load_modelpath,
		is_distribute=torch.cuda.device_count() > 1,
		src_classes=64,
		cos_fac=params.cosine_fac,
	)
	
	# contrastive finetune
	finetune(source_loader, novel_loader, total_epoch, model_params, dataloader_params, params)
	
	# save results
	consolidate_results()
