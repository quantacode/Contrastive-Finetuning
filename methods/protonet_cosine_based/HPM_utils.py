import ipdb
import torch
import numpy as np
from scipy import stats

def get_hp_indices(affinity, numpos, hpm_type):
	if hpm_type=='t1':
		assert numpos==1
		# nearest positive
		affinity = affinity.mean(dim=1).numpy()
		return_idx = list(np.argsort(affinity)[-numpos:])
	elif hpm_type=='t2':
		# sample from distribution over "closest in each view" (if occurrence>1)
		assert numpos==1
		# degree of uniformity of distribution
		alpha = 1.0
		sorted_aff = affinity.argsort(dim=0)
		closest_indices = sorted_aff[-numpos]
		unique_indices, counts = closest_indices.unique(return_counts=True)
		# make sure nearest pos occurs in more than 1 view, otherwise leave untouched
		frequent_mask = torch.where(counts>1)
		if len(frequent_mask[0])>0:
			unique_indices, counts = unique_indices[frequent_mask], counts[frequent_mask]
		counts = torch.exp(counts/alpha)
		prob = counts/counts.sum()
		sampled_idx = torch.multinomial(prob, 1, replacement=True)
		return_idx = [int(unique_indices[sampled_idx].item())]
	elif hpm_type=='t3':
		# sample from distribution over "k nearest in each view" (if occurrence>1)
		assert numpos==1
		# degree of uniformity of distribution
		k = 3
		sorted_aff = affinity.argsort(dim=0)
		closest_indices = sorted_aff[-numpos-(k-1):].view(-1)
		unique_indices, counts = closest_indices.unique(return_counts=True)
		# make sure nearest pos occurs in more than 1 view, otherwise leave untouched
		frequent_mask = torch.where(counts>1)
		if len(frequent_mask[0])>0:
			unique_indices, counts = unique_indices[frequent_mask], counts[frequent_mask]
		counts = torch.exp(counts/0.5)
		prob = counts/counts.sum()
		sampled_idx = torch.multinomial(prob, 1, replacement=True)
		return_idx = [int(unique_indices[sampled_idx].item())]

	return return_idx