# Contastive-ft + CE-source-ft

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
from itertools import combinations

import configs
from options import parse_args, get_best_file, get_assigned_file
from methods.backbone import model_dict
from data.datamgr import SimpleDataManager
from data.cssf_datamgr_custom_collate import SetDataManager, ContrastiveBatchifier
from data import get_datafiles
from methods.weight_imprint_based.ConCE import ConCeModel

from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from shutil import copy
import pickle
import ipdb

def record_each(q_type, quantity, results_path):
    for key in quantity.keys():
        # for each epoch
        if len(quantity[key])!=0:
            quantity_this = quantity[key]
            quantity_this = np.asarray(quantity_this)
            with open(results_path.replace('results.txt', '%s%s.pkl'%(q_type,key)), 'wb') as f:
                pickle.dump(quantity_this, f)

def record(acc_all_le, acc_all, accDiff_all, results_path):
    acc_all_le = np.asarray(acc_all_le)

    with open(results_path.replace('results.txt', 'linev.pkl'), 'wb') as f:
        pickle.dump(acc_all_le, f)

    # accuracy
    record_each('wi_final', acc_all, results_path)
    record_each('wi_delta', accDiff_all, results_path)

def get_init_var():
    acc_all_le = []
    acc_all = {'0':[]}
    accDiff_all = {'0':[]}
    return acc_all_le, acc_all, accDiff_all

def finetune(source_loader, novel_loader, total_epoch, model_params, dataloader_params, params):
    results_dir = params.results_dir
    n_way, n_support = dataloader_params['n_way'], dataloader_params['n_support']
    image_size = dataloader_params['image_size']
    acc_all_le, acc_all, accDiff_all = get_init_var()

    # model
    model = ConCeModel(n_way=n_way, projhead=params.projection_head,
                       ft_mode=params.ft_mode, **model_params)

    if params.outfile:
        jobdir = createdir(model_params['tf_path'].replace('log', 'job'))
        jobfile = os.path.join(jobdir, 'outfile.txt')
        print('---------- training progress file -----------\n%s' % (jobfile))
        fileout = open(jobfile, 'w')
        progress = tqdm(novel_loader, file=fileout, dynamic_ncols=True)
    else:
        progress = tqdm(novel_loader)

    for task_id, (x, x_ft, y) in enumerate(progress):
        if params.model=='resnet12':
            supcon_datamgr = ContrastiveBatchifier(n_way=n_way, n_support=n_support, image_size=image_size,
                                                   augstrength=params.augstrength)
            if params.is_tgt_aug:
                supcon_dataloader = supcon_datamgr.get_loader([sample[:n_support] for sample in x_ft])
            else:
                supcon_dataloader = None

            x_supp, _ = next(iter(supcon_dataloader))
            x_supp = x_supp.contiguous().view(n_way, n_support, *x_supp.shape[1:])
            x_fs = torch.cat([x_supp, x[:,n_support:,:,:,:]],dim=1)
        else:
            x_fs = x
        ###############################################################################################
        # load pretrained feature extractor
        model.refresh_from_chkpt()
        model.cuda()
        n_query = x.size(1) - n_support
        x_fs = x_fs.cuda()
        acc_before, _, z_all_analysis = model.validate(n_way, n_support, n_query, x_fs, -1)
        acc_all[str(0)].append(acc_before)
        improvement = acc_before - acc_before
        accDiff_all[str(0)].append(improvement)

        acc_all_le.append(0)
        if (task_id+1)%10==0 or task_id==len(novel_loader)-1:
            record(acc_all_le, acc_all, accDiff_all,
                   results_path=os.path.join(results_dir,'results.txt'))

    if params.outfile: fileout.close()

if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train/ufsl/cssf/parallel')
    print(params)
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
    if 'Conv' in params.model or params.model=='resnet12':
        if params.targetset == 'omniglot':
            image_size = 28
        else:
            image_size = 84
    else:
        image_size = 224
    num_tasks = params.num_tasks
    total_epoch = params.stop_epoch

    # dataloaders
    target_file, target_val_file, target_novel_file = get_datafiles('cssf', params, configs)
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

    if params.ufsl_dataset:
        from data.cssf_datamgr_ufsl import ContrastiveBatchifier
        from data.datamgr_ufsl import SimpleDataManager

        if params.targetset == 'celeba':
            from data.cssf_datamgr_custom_collate import CelebaDM as SetDataManager
        else:
            from data.cssf_datamgr_custom_collate import SetDataManager
        novel_loader = SetDataManager(**dataloader_params).get_data_loader(inference_file)
        source_loader = SimpleDataManager(image_size, params.n_unlab_neg, dataset=params.targetset).get_data_loader(
            target_file, aug=params.is_src_aug)
    else:
        if params.model=='resnet12': # rfs comparison
            from data.cssf_datamgr_rfs import SetDataManager, ContrastiveBatchifier
        else:
            from data.cssf_datamgr_custom_collate import SetDataManager, ContrastiveBatchifier

        novel_loader = SetDataManager(**dataloader_params).get_data_loader(inference_file)

    print('\n------------ validating : %s --------------'%(params.hyperparam_select))
    print('method : ConCE mode (%s)'%params.ft_mode)
    print('source file: %s'%(target_file))
    print('novel file: %s'%(inference_file))
    print('novel loader file: %s'%(novel_loader.dataset.data_file))
    print('distractor sz: %s'%(params.n_unlab_neg))
    print('tau : %s'%(params.tau))
    if params.ft_mode == 'rw_lpan':
        print('alpha : %s'%(params.alpha))
    print('lr : %s'%(params.lr))
    print('Source Aug : ', params.is_src_aug)
    print('Target Aug : ', params.is_tgt_aug)
    print('Novel Shots : %d\n'%(params.n_shot))

    # model
    model_params = dict(
        model_func=model_dict[params.model],
        tau=params.tau,
        tf_path=params.tf_dir,
        loadpath=params.load_modelpath,
        is_distribute=torch.cuda.device_count() > 1,
        src_classes=64,
        cos_fac=params.cosine_fac
    )
    source_loader=None
    finetune(source_loader, novel_loader, total_epoch, model_params, dataloader_params, params)

