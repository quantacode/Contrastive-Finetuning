import torch
import numpy as np
import os
import json
import ipdb
import math
import pickle
import matplotlib.pyplot as plt


def one_hot(y, num_class):
	return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)


def DBindex(cl_data_file):
	class_list = cl_data_file.keys()
	cl_num = len(class_list)
	cl_means = []
	stds = []
	DBs = []
	for cl in class_list:
		cl_means.append(np.mean(cl_data_file[cl], axis=0))
		stds.append(np.sqrt(np.mean(np.sum(np.square(cl_data_file[cl] - cl_means[-1]), axis=1))))

	mu_i = np.tile(np.expand_dims(np.array(cl_means), axis=0), (len(class_list), 1, 1))
	mu_j = np.transpose(mu_i, (1, 0, 2))
	mdists = np.sqrt(np.sum(np.square(mu_i - mu_j), axis=2))

	for i in range(cl_num):
		DBs.append(np.max([(stds[i] + stds[j]) / mdists[i, j] for j in range(cl_num) if j != i]))
	return np.mean(DBs)


def sparsity(cl_data_file):
	class_list = cl_data_file.keys()
	cl_sparsity = []
	for cl in class_list:
		cl_sparsity.append(np.mean([np.sum(x != 0) for x in cl_data_file[cl]]))

	return np.mean(cl_sparsity)


def createdir(savedir):
	if not os.path.isdir(savedir): os.makedirs(savedir)
	return savedir


def get_miniImagenet_labelnames(labelnames):
	with open('/home/rajshekd/projects/FSG/FSG_raj/cdfsl/filelists/miniImagenet/classnames.txt') as f:
		lines = f.readlines()
	classdict = {}
	for line in lines:
		line = line.split('\n')[0]
		if line == '#### Val ####':
			break
		key, name = line.split(' ')
		classdict[key] = name
	aliases = []
	for lname in labelnames:
		aliases.append(classdict[lname])
	return aliases


def json_dump(obj, filename):
	with open(filename, 'w') as f:
		json.dump(obj, f)


def pickle_dump(filename, obj):
	with open(filename, 'wb') as f:
		pickle.dump(obj, f)


def json_load(filename):
	with open(filename) as f:
		obj = json.load(f)
	return obj


def pickle_load(filename):
	with open(filename, 'rb') as f:
		obj = pickle.load(f)
	return obj


def classwise_affinity_graph(affinity_graph, labels, c1, c2):
	uL = np.unique(labels)
	label1 = uL[c1]
	label2 = uL[c2]
	c1_indices = np.asarray([i for i, lab in enumerate(labels) if lab == label1])
	c2_indices = np.asarray([i for i, lab in enumerate(labels) if lab == label2])
	return affinity_graph[c1_indices[:, None], c2_indices]


def aggregate_accuracy(test_logits_sample, test_labels):
	"""
	Compute classification accuracy.
	"""
	averaged_predictions = torch.logsumexp(test_logits_sample, dim=0)
	return torch.mean(torch.eq(test_labels, torch.argmax(averaged_predictions, dim=-1)).float())


def mat_sigmoid(mat):
	return 1 / (1 + np.exp(-mat))


def mat_normalize(mat):
	return (mat - mat.min()) / (mat.max() - mat.min())


def preprocess_image(x):
	img = x.permute(1, 2, 0).cpu().numpy()
	img = (img - img.min()) / (img.max() - img.min())
	return img


def save_image(x, name):
	plt.imsave(name, preprocess_image(x))


def chkpt_vis(model, n1, n2):
	# for i, (name, val) in enumerate(list(model.named_parameters())[n1:n2]):
	#     print(name, ': ', val.mean().item())
	# print('-------------------')
	for i, (name, val) in enumerate(list(model.state_dict().items())[n1:n2]):
		print(name, ': ', val.float().mean().item())
	print('-------------------')
