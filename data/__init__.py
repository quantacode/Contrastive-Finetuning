import ipdb
import os

def get_datafiles(params, configs):
	target_file = params.distractor_set
	target_val_file = os.path.join(configs.data_dir[params.targetset], 'val.json')
	target_novel_file = os.path.join(configs.data_dir[params.targetset], 'novel.json')
	return target_file, target_val_file, target_novel_file