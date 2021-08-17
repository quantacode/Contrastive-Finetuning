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
	root = '/jet/home/rajshekd/projects/PRALIGN/CSSF/output2/cssf_WIEval_1shot/ewn_lpan/testing/arch-ResNet10_cosineLC_Pretrain-Src/cars-tgt_mIN-src/augSrc-False_augTgt-True/neg-L16U128_ftEP-600_lr-0.005_tau-0.05'
	NE_list = [25,50,100,200]
	# NE_list = range(0, 605, 5)
	max_parallel_id = 29
	with open(os.path.join(root, 'analysis2.txt'), 'w') as f:
		f.write("support spread\n"
		        "support sep\n"
		        "query spread\n"
		        "query sep\n")

		for ine, num_epochs in enumerate(NE_list):
			support_spread_all = []
			support_spread_pcnt_all = []
			support_sep_all = []
			support_sep_pcnt_all = []
			query_spread_all = []
			query_spread_pcnt_all = []
			query_sep_all = []
			query_sep_pcnt_all = []
			# flag=1
			chklist = []
			for run in range(0,max_parallel_id+1):
				final_name = os.path.join(root, str(run), 'results/wi_final%d.pkl'%(num_epochs))
				if not os.path.exists(final_name):
					chklist.append(run)
					continue
				# if not os.path.exists(final_name):
				# flag =-1
				# break
				support_cspread_name = os.path.join(root, str(run), 'results/wi_support_cspread%d.pkl'%(num_epochs))
				support_cspread_pcnt_name = os.path.join(root, str(run), 'results/wi_support_cspread_pcnt%d.pkl'%(num_epochs))
				support_csep_name = os.path.join(root, str(run), 'results/wi_support_csep%d.pkl'%(num_epochs))
				support_csep_pcnt_name = os.path.join(root, str(run), 'results/wi_support_csep_pcnt%d.pkl'%(num_epochs))

				support_spread_all = append_data(support_cspread_name, support_spread_all)
				support_spread_pcnt_all = append_data(support_cspread_pcnt_name, support_spread_pcnt_all)
				support_sep_all = append_data(support_csep_name, support_sep_all)
				support_sep_pcnt_all = append_data(support_csep_pcnt_name, support_sep_pcnt_all)

				query_cspread_name = os.path.join(root, str(run), 'results/wi_query_cspread%d.pkl'%(num_epochs))
				query_cspread_pcnt_name = os.path.join(root, str(run), 'results/wi_query_cspread_pcnt%d.pkl'%(num_epochs))
				query_csep_name = os.path.join(root, str(run), 'results/wi_query_csep%d.pkl'%(num_epochs))
				query_csep_pcnt_name = os.path.join(root, str(run), 'results/wi_query_csep_pcnt%d.pkl'%(num_epochs))

				query_spread_all = append_data(query_cspread_name, query_spread_all)
				query_spread_pcnt_all = append_data(query_cspread_pcnt_name, query_spread_pcnt_all)
				query_sep_all = append_data(query_csep_name, query_sep_all)
				query_sep_pcnt_all = append_data(query_csep_pcnt_name, query_sep_pcnt_all)

			# if flag==-1:
			# 	continue
			if len(query_sep_all) == 0:
				continue
			print(num_epochs)
			print('KILLED TASK IDS: ', chklist)

			support_sep_all = np.hstack(support_sep_all)
			support_sep_pcnt_all = np.hstack(support_sep_pcnt_all)
			support_sep_before = support_sep_all/support_sep_pcnt_all

			support_spread_all = np.hstack(support_spread_all)
			# support_spread_pcnt_all = np.hstack(support_spread_pcnt_all)
			support_spread_pcnt_all = support_spread_all/support_sep_before

			query_sep_all = np.hstack(query_sep_all)
			# query_sep_pcnt_all = np.hstack(query_sep_pcnt_all)
			query_sep_pcnt_all = query_sep_all/support_sep_before

			query_spread_all = np.hstack(query_spread_all)
			# query_spread_pcnt_all = np.hstack(query_spread_pcnt_all)
			query_spread_pcnt_all = query_spread_all/support_sep_before

			nTasks = support_spread_all.shape[0]
			support_spread_mean = np.mean(support_spread_all)
			support_spread_pcnt_mean = np.mean(support_spread_pcnt_all)
			support_sep_mean = np.mean(support_sep_all)
			support_sep_pcnt_mean = np.mean(support_sep_pcnt_all)
			query_spread_mean = np.mean(query_spread_all)
			query_spread_pcnt_mean = np.mean(query_spread_pcnt_all)
			query_sep_mean = np.mean(query_sep_all)
			query_sep_pcnt_mean = np.mean(query_sep_pcnt_all)

			# f.write("ep= %d \n"
			f.write(
			        "%4.2f\t%4.2f\n"
			        "%4.2f\t%4.2f\n" % (
				        support_spread_pcnt_mean, query_spread_pcnt_mean,
				        support_sep_pcnt_mean, query_sep_pcnt_mean))
		# f.write("ep= %d \n"
		# 	        "support spread\n"
		# 	        "support sep\n"
		# 	        "query spread\n"
		# 	        "query sep\n"
		# 	        "%4.2f (%4.2f) \n"
		# 	        "%4.2f (%4.2f) \n"
		# 	        "%4.2f (%4.2f) \n"
		# 	        "%4.2f (%4.2f) \n\n" % (num_epochs,
		# 		support_spread_mean, support_spread_pcnt_mean, support_sep_mean, support_sep_pcnt_mean,
		# 		query_spread_mean, query_spread_pcnt_mean, query_sep_mean, query_sep_pcnt_mean))
		f.write(" nTasks:%d\n\n"%(nTasks))
