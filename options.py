import numpy as np
import os
import glob
import torch
import argparse
import ipdb

def parse_args(script):
  parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
  parser.add_argument('--dataset', default='multi', help='miniImagenet/cub/cars/places/plantae, specify multi for training with multiple domains')
  parser.add_argument('--testset', default='cub', help='cub/cars/places/plantae, valid only when dataset=multi')
  parser.add_argument('--valset', default='cub', help='cub/cars/places/plantae, valid only when dataset=multi')
  parser.add_argument('--model', default='ResNet10', help='model: Conv{4|6} / ResNet{10|18|34}')
  parser.add_argument('--method-type', default='baseline',help='baseline/baseline++/protonet/matchingnet/relationnet{_softmax}/gnnnet')
  parser.add_argument('--train_n_way' , default=5, type=int,  help='class num to classify for training')
  parser.add_argument('--test_n_way'  , default=5, type=int,  help='class num to classify for testing (validation) ')
  parser.add_argument('--n_shot'      , default=5, type=int,  help='number of labeled data in each class, same as n_support')
  parser.add_argument('--n_query'      , default=15, type=int,  help='num queries')
  parser.add_argument('--train_aug'   , action='store_true',  help='perform data augmentation or not during training ')
  parser.add_argument('--debug'   , action='store_true',  help='debug')
  parser.add_argument('--name'        , default='tmp', type=str, help='')
  parser.add_argument('--save_dir'    , default='experiments', type=str, help='directory for logs and checkpoints')
  parser.add_argument('--data-dir'    , default='./filelists', type=str, help='')
  parser.add_argument('--image-size', default=84, type=int, help='tUn vs Semi supervised')
  parser.add_argument('--load-modelpath', default=None, type=str, help='')
  parser.add_argument('--target-datapath', default=None, help='if specific name for target path')
  parser.add_argument('--augstrength', default='0', type=str, help='level of augmentation')
  parser.add_argument('--num_classes', default=200, type=int,
                      help='total number of classes in softmax, only used in baseline')
  parser.add_argument('--freeze_backbone', action='store_true',
                      help='perform data augmentation or not during training ')
  if 'train' in script:
    parser.add_argument('--save_freq'   , default=25, type=int, help='Save frequency')
    parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
    parser.add_argument('--stop_epoch'  , default=100000000, type=int, help ='Stopping epoch')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--resume'      , default='', type=str, help='continue from previous trained model with largest epoch')
    parser.add_argument('--resume_epoch', default=-1, type=int, help='')
    parser.add_argument('--warmup'      , default='gg3b0', type=str, help='continue from baseline, neglected if resume is true')
    parser.add_argument('--Nmb', default=1, type=int, help='num episodes per batch')
    parser.add_argument('--hyperparam_select', action='store_true', help='hyperparameter selection using val split')
    parser.add_argument("--subpix_pcnt", type=float, default=0.2, help="top score pcnt for dense protonet.")
    if 'contrastive' in script:
      parser.add_argument('--targetset', default='cub', help='for adaptation')
      parser.add_argument('--temperature', default=0.5, type=float, help='contrastive loss temperature')
      parser.add_argument('--contrastive-wt', default=1.0, type=float, help='contrastive loss temperature')
      parser.add_argument('--contrastive-batch-size', default=-1, type=int, help='feature batchsz')
    elif 'clusterReg' in script:
      parser.add_argument('--targetset', default='cub', help='for adaptation')
      parser.add_argument('--cr-wt', default=1.0, type=float, help='contrastive loss temperature')
      parser.add_argument('--cr-batch-size', default=-1, type=int, help='feature batchsz')
      parser.add_argument('--num-clusters', default=200, type=int, help=' num clusters')
      parser.add_argument('--sim-bias', default=0.5, type=float, help=' similarity bias')
    elif 'ufsl' in script:
      parser.add_argument('--targetset', default='cub', help='for adaptation')
      parser.add_argument('--ufsl-wt', default=1.0, type=float, help='contrastive loss temperature')
      parser.add_argument('--dominant-id', default=0, type=int, help='redefine sampling based on gt classes, only for '
                                                                     'debugging')
      parser.add_argument('--dominant-p', default=0.9, type=float, help='only for debugging')
      parser.add_argument('--projection-head', action='store_true', help='proj head on top of features')
      if 'l1' in script:
        parser.add_argument('--l1reg', action='store_true', help='to include l1 reg')
        parser.add_argument('--tgt-batch-size', default=-1, type=int, help='target batchsz')
        parser.add_argument('--reg-wt', default=1.0, type=float, help='regularization wt')
        parser.add_argument('--tau-l1', default=1.0, type=float, help='similarity function param')
      elif ('mvcon' in script) or ('npcon' in script):
        parser.add_argument('--tau', default=0.1, type=float, help='temperature')
        parser.add_argument('--latentcls_prob', type=float, help='latent class prob')
        parser.add_argument('--modification', default='none', type=str, help='none/debiased/debiased_hpm')
        parser.add_argument('--pos-wt-type', default='uniform', type=str, help='weighting schemes for positives')
        parser.add_argument('--mvcon-type', default='vanila', type=str, help='vanila: supcon, plus: remove positves '
                                                                             'from denominator')
        parser.add_argument('--num_pos', default=-1, type=int, help='')
        parser.add_argument('--load-modelpath-aux', default=None, type=str, help='')
        parser.add_argument('--hpm_type', default=None, type=str, help='type of hardpositive mining')
      elif 'hpm' in script:
        parser.add_argument('--beta', default=0.5, type=float, help='temperature')
        parser.add_argument('--tau', default=0.5, type=float, help='temperature')
      elif 'cssf' in script:
        parser.add_argument('--tau', default=0.5, type=float, help='temperature')
        parser.add_argument('--cosine_fac', default=1.0, type=float, help='temperature for cosine classifier')
        parser.add_argument('--alpha', default=0.5, type=float, help='hnm+align convex parameter')
        parser.add_argument('--bd_alpha', default=0.5, type=float, help='hnm+align convex parameter') 
        parser.add_argument('--beta', default=1.0, type=float, help='hpm parameter')
        parser.add_argument('--clstau', default=1.0, type=float, help='dataset dependent parameter')
        parser.add_argument('--distractor_bsz', default=64, type=int, help='distractor batch size')
        parser.add_argument('--src_subset_sz', default=64, type=int, help='source batch size')
        parser.add_argument('--src_classes', default=64, type=int, help='source classifier size')
        parser.add_argument('--num_tasks', default=600, type=int, help='num_tasks')
        parser.add_argument('--num_ft_layers', default=0, type=int, help='# finetuning layers')
        parser.add_argument('--is_same_head', action='store_true', help='same projection head for OL and LPUN')
        parser.add_argument('--is_src_aug', action='store_true', help='augmentation to source samples')
        parser.add_argument('--is_tgt_aug', action='store_true', help='augmentation to target (novel) samples')
        parser.add_argument('--ufsl_dataset', action='store_true', help='if ufsl experiments')
        parser.add_argument('--ceft', action='store_true', help='replacing contr. with ce classifier in conce')
        parser.add_argument('--ft_mode', default='preFT', type=str, help='cssf finetuning type')
        parser.add_argument('--distractor_set', default='./filelists/miniImagenet/base.json', type=str, help='distractor dataset')
        if 'parallel' in script:
          parser.add_argument('--run_id', default=0, type=int, help='parallel process id')


    elif 'transfer_mvcon' in script:
      parser.add_argument('--src_n_way', default=5, type=int, help='class num to classify for training')
      parser.add_argument('--src_n_query', default=5, type=int, help='class num to classify for training')
      parser.add_argument('--targetset', default='cub', help='for adaptation')
      parser.add_argument('--featype', default='projection', help='feature ex for source fsl')
      parser.add_argument('--projtype', default='same', help='projection for supcon')
      parser.add_argument('--ufsl-wt', default=1.0, type=float, help='contrastive loss temperature')
      parser.add_argument('--tau', default=0.5, type=float, help='temperature')
      parser.add_argument('--pos-wt-type', default='min', type=str, help='weighting schemes for positives')
      parser.add_argument('--batch_size', default=-1, type=int, help='target batchsz')
      parser.add_argument('--mvcon-type', default='vanila', type=str, help='vanila: supcon, plus: remove positves '
                                                                           'from denominator')
    elif ('transfer_Contrastive' in script) or ('transfer_simclr' in script):
      parser.add_argument('--targetset', default='cub', help='for adaptation')
      parser.add_argument('--ufsl-wt', default=1.0, type=float, help='contrastive loss temperature')
      parser.add_argument('--tau', default=0.5, type=float, help='temperature')
      parser.add_argument('--batch-size', default=-1, type=int, help='target batchsz')
      parser.add_argument('--num_pos', default=-1, type=int, help='')
      parser.add_argument('--hpm_type', default=None, type=str, help='type of hardpositive mining')

    elif 'interp' in script:
      parser.add_argument('--sampling', default='random', help='support and query sampling')
      parser.add_argument('--targetset', default='cub', help='for adaptation')
      parser.add_argument('--ufsl-wt', default=1.0, type=float, help='contrastive loss temperature')
      parser.add_argument('--num-aug', default=4, type=int, help='')
  elif script == 'test':
    parser.add_argument('--split'       , default='novel', help='base/val/novel')
    parser.add_argument('--save_epoch', default=-1, type=int,help ='save feature from the model trained in x epoch, '
                                                                  'use the best model if x is -1')
    parser.add_argument('--batch-size', default=-1, type=int, help='feature batchsz')
  elif script == 'cluster':
    parser.add_argument('--ss-thresh', default=0.99, type=float, help='self similarity threshold')
    parser.add_argument('--min-shots', default=0, type=int, help='num shots per class')
    parser.add_argument('--split'       , default='novel', help='base/val/novel')
    parser.add_argument('--num-classes' , default=200, type=int, help=' num clusters')
    parser.add_argument('--relabel', action='store_true', help='tUn vs Semi supervised')
    parser.add_argument('--batch-size', default=-1, type=int, help='feature batchsz')
    parser.add_argument('--tau', default=0.1, type=float, help='feature batchsz')
    parser.add_argument('--data-dir-save', default='./filelists', type=str, help='')
    parser.add_argument('--save_epoch', default=-1, type=int,help ='save feature from the model trained in x epoch, '
                                                                  'use the best model if x is -1')
  else:
    raise ValueError('Unknown script')

  return parser.parse_args()

def get_assigned_file(checkpoint_dir,num):
  assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
  return assign_file

def get_resume_file(checkpoint_dir, resume_epoch=-1):
  filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
  if len(filelist) == 0:
    return None

  filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
  epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
  max_epoch = np.max(epochs)
  epoch = max_epoch if resume_epoch == -1 else resume_epoch
  resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(epoch))
  return resume_file

def get_best_file(checkpoint_dir):
  best_file = os.path.join(checkpoint_dir, 'best_model.tar')
  if os.path.isfile(best_file):
    return best_file
  else:
    return get_resume_file(checkpoint_dir)

def load_warmup_state(filename, method):
  print('  load pre-trained model file: {}'.format(filename))
  # warmup_resume_file = get_resume_file(filename)
  warmup_resume_file = get_best_file(filename)
  tmp = torch.load(warmup_resume_file)
  if tmp is not None:
    state = tmp['state']
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
      if 'relationnet' in method and "feature." in key:
        newkey = key.replace("feature.","")
        state[newkey] = state.pop(key)
      elif method == 'gnnnet':
        if 'feature.module.' in key:
          if (torch.cuda.device_count() > 1):
            newkey = key.replace("feature.module.", "module.")
          else:
            newkey = key.replace("feature.module.", "")
          state[newkey] = state.pop(key)
        elif 'feature.' in key:
          newkey = key.replace("feature.", "")
          state[newkey] = state.pop(key)
      elif method == 'matchingnet' and 'feature.' in key and '.7.' not in key:
        newkey = key.replace("feature.","")
        state[newkey] = state.pop(key)
      else:
        state.pop(key)
  else:
    raise ValueError(' No pre-trained encoder file found!')
  return state

