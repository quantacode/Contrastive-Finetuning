from . import datamgr
from . import dataset
from . import additional_transforms
from . import feature_loader
import ipdb
import os
def get_datafiles(type, params, configs):
	if type == 'umtra' or type=='mvcon':
		if params.targetset == 'miniImagenet' or params.targetset == 'omniglot' :
			print('\n--- prepare dataloader ---')
			target_file = os.path.join(configs.data_dir[params.targetset], 'base.json')
			target_val_file = os.path.join(configs.data_dir[params.targetset], 'val.json')
			target_novel_file = os.path.join(configs.data_dir[params.targetset], 'novel.json')
		elif params.targetset == 'celeba':
			print('\n--- prepare dataloader ---')
			target_file = os.path.join(configs.data_dir[params.targetset], 'all.json')
			target_val_file = target_file
			target_novel_file = target_file
		else:
			print('\n--- prepare dataloader ---')
			target_file = os.path.join(configs.data_dir[params.targetset], 'practice_train.json')
			target_val_file = os.path.join(configs.data_dir[params.targetset], 'practice_val.json')
			target_novel_file = os.path.join(configs.data_dir[params.targetset], 'novel.json')
		return target_file, target_val_file, target_novel_file
	elif type=='cssf':
		if params.targetset == 'miniImagenet' or params.targetset == 'omniglot' :
			target_file = os.path.join(configs.data_dir[params.targetset], 'base.json')
		elif params.targetset == 'celeba':
			target_file = os.path.join(configs.data_dir[params.targetset], 'all.json')
		else:
			target_file = params.unlab_split
		target_val_file = os.path.join(configs.data_dir[params.targetset], 'val.json')
		target_novel_file = os.path.join(configs.data_dir[params.targetset], 'novel.json')
		return target_file, target_val_file, target_novel_file
	elif type == 'src-baseline' or 'src-protonet' in type or 'src-gnnnet' in type:
		if params.dataset == 'multi':
			print('  train with multiple seen domains (unseen domain: {})'.format(params.testset))
			datasets = ['miniImagenet', 'cars', 'places', 'cub', 'plantae']
			datasets.remove(params.testset)
			base_file = [os.path.join(params.data_dir, dataset, 'base.json') for dataset in datasets]
			val_file = os.path.join(params.data_dir, 'miniImagenet', 'val.json')
		elif params.dataset == 'multi_finegrained':
			print('  train with multiple seen domains')
			datasets = ['aircraft', 'vgg_flower', 'plantae']
			base_file = [os.path.join(params.data_dir, dataset, 'base.json') for dataset in datasets]
			datasets.append(params.testset)
			val_file = [(dataset, os.path.join(params.data_dir, dataset, 'val.json')) for dataset in datasets]
		else:
			print('  train with single seen domain {}'.format(params.dataset))
			base_file = os.path.join(params.data_dir, params.dataset, 'base.json')
			val_file = os.path.join(params.data_dir, params.dataset, 'val.json')
		return base_file, val_file
	elif type == 'src+tgt' in type:
		print('\n--- prepare dataloader ---')
		base_file = os.path.join(params.data_dir, params.dataset, 'base.json')
		val_file = os.path.join(params.data_dir, params.dataset, 'val.json')
		if params.targetset == 'miniImagenet':
			target_file = os.path.join(configs.data_dir[params.targetset], 'base.json')
			target_val_file = os.path.join(configs.data_dir[params.targetset], 'val.json')
		else:
			target_file = os.path.join(configs.data_dir[params.targetset], 'practice_train.json')
			target_val_file = os.path.join(configs.data_dir[params.targetset], 'practice_val.json')
		target_novel_file = os.path.join(configs.data_dir[params.targetset], 'novel.json')
		return base_file, val_file, target_file, target_val_file, target_novel_file