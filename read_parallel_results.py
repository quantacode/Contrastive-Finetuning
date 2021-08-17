import os
import pickle
import numpy as np
import ipdb

def append_data(name, quantity):
        with open(name, 'rb') as f:
                data = pickle.load(f)
                quantity.append(data)
        return quantity
if __name__=="__main__":
	root = 'output2/cssf_WIEval_5shot/ewn_lpan/testing/arch-Conv4_medium_cosineLC_Pretrain-Src/miniImagenet-tgt_mIN-src/augSrc-False_augTgt-True/neg-L16U64_ftEP-400_lr-0.0005_tau-0.05'
	#NE_list = [10, 20, 40, 60, 80, 100]
	NE_list = range(0,600,10)
	max_parallel_id = 6
	with open(os.path.join(root, 'results.txt'), 'w') as f:
		for ine, num_epochs in enumerate(NE_list):
			acc_all_le = []
			acc_all = []
			accDiff_all = []

			# flag=1
			# collect parallel results
			chklist = []
			for run in range(0,max_parallel_id+1):
				linev_name = os.path.join(root, str(run), 'results/linev.pkl')
				final_name = os.path.join(root, str(run), 'results/wi_final%d.pkl'%(num_epochs))
				if not os.path.exists(final_name):
					chklist.append(run)
					continue
				# if not os.path.exists(final_name):
					# flag =-1
					# break
				delta_name = os.path.join(root, str(run), 'results/wi_delta%d.pkl'%(num_epochs))

				acc_all_le = append_data(linev_name, acc_all_le)
				acc_all = append_data(final_name, acc_all)
				accDiff_all = append_data(delta_name, accDiff_all)

			# if flag==-1:
			# 	continue
			if len(acc_all_le)==0:
				continue
			print(num_epochs)
			print('KILLED TASK IDS: ', chklist)
			acc_all_le = np.hstack(acc_all_le)
			acc_all = np.hstack(acc_all)
			accDiff_all = np.hstack(accDiff_all)

			nTasks = acc_all_le.shape[0]
			acc_mean_le = np.mean(acc_all_le)
			acc_std_le = 1.96 * np.std(acc_all_le) / np.sqrt(nTasks)
			acc_mean = np.mean(acc_all)
			acc_std = 1.96 * np.std(acc_all) / np.sqrt(nTasks)
			accDiff_mean = np.mean(accDiff_all)
			accDiff_std = 1.96 * np.std(accDiff_all) / np.sqrt(nTasks)

			f.write("ep= %d \n"
			        "acc_final (rel_gain):\n"
			        "%4.2f +- %4.2f \n"
			        "%4.2f +- %4.2f \n\n" % (
				num_epochs, acc_mean, acc_std, accDiff_mean, accDiff_std))
		f.write("nTasks:%d\n" % (nTasks))

