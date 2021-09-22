import os
import pickle
import numpy as np
import ipdb


def append_data(name, quantity):
	with open(name, 'rb') as f:
		data = pickle.load(f)
		quantity.append(data)
	return quantity


def consolidate_results(root, max_parallel_id=0):
	NE_list = range(0, 600, 10)
	with open(os.path.join(root, 'results.txt'), 'w') as f:
		for ine, num_epochs in enumerate(NE_list):
			acc_all_le = []
			acc_all = []
			accDiff_all = []
			chklist = []
			for run in range(0, max_parallel_id + 1):
				linev_name = os.path.join(root, str(run), 'results/linev.pkl')
				final_name = os.path.join(root, str(run), 'results/wi_final%d.pkl' % (num_epochs))
				if not os.path.exists(final_name):
					chklist.append(run)
					continue
				delta_name = os.path.join(root, str(run), 'results/wi_delta%d.pkl' % (num_epochs))
				
				acc_all_le = append_data(linev_name, acc_all_le)
				acc_all = append_data(final_name, acc_all)
				accDiff_all = append_data(delta_name, accDiff_all)
			
			if len(acc_all_le) == 0:
				continue
			acc_all_le = np.hstack(acc_all_le)
			acc_all = np.hstack(acc_all)
			
			nTasks = acc_all_le.shape[0]
			acc_mean = np.mean(acc_all)
			acc_std = 1.96 * np.std(acc_all) / np.sqrt(nTasks)
			
			strn = "accuracy = %4.2f +- %4.2f" % (acc_mean, acc_std)
		print(strn)


if __name__ == "__main__":
	consolidate_results(root)
