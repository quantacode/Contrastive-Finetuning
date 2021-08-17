#!/bin/bash
# sbatch -p GPU-shared --gpus=1 -N 1 -t 20:00:00 --ntasks-per-node=5 --mail-type=ALL
# sbatch -p GPU-shared --gpus=1 -N 1 -t 1:00:00 --ntasks-per-node=5 --mail-type=ALL

#echo commands to stdout
set -x
cd /jet/home/rajgpu/projects/PRALIGN/CSSF
#module load anaconda3
source activate py36

##############################################################

# INDEX
# OL - only labelled contrastive loss
# CE - standard CE classfier
# LPAN - labelled positives and all negative contrastive loss

# # ######################## CONFT #####################################
# AVAILABLE MODES: (cb is supcon implemetation i.e. complete batch of positives)
# lpan, lpan_cb, lpun, lpun, lpan_cb_supcon, lpan_mtce (conce), lpan_mtce_cb, ce_mtce

# # ~~~~~ lpan ~~~~~
# CUDA_VISIBLE_DEVICES=0  python finetune_ConCe_parallelized.py \
# --ft_mode 'lpan' \
# --is_tgt_aug --n_shot 1 \
# --train_aug --method 'weight_imprint' --model 'ResNet10' \
# --stop_epoch 400 --num_tasks 20 --lr 0.005 \
# --targetset cars --tau 0.05 --n_unlab_neg 128 \
# --name 'cssf_WIEval_1shot/lpan/testing/arch-ResNet10_cosineLC_Pretrain-Src/cars-tgt_mIN-src/augSrc-False_augTgt-True/neg-L16U128_ftEP-400_lr-0.005_tau-0.05' \
# --load-modelpath 'output2/miniImagenet_ResNet10_lft/399.tar' \
# --unlab_split './filelists/miniImagenet/base.json' \
# --outfile \
# --run_id $1
# # --projection-head


# ~~~~~ MAIN ewn (equally weighted task and distractors negatives) ~~~~~
CUDA_VISIBLE_DEVICES=0  python finetune_ConCe_parallelized.py \
--ft_mode 'ewn_lpan' \
--is_tgt_aug --n_shot 1 \
--train_aug --method 'weight_imprint' --model 'resnet12' \
--stop_epoch 300 --num_tasks 20  --lr 0.0001 \
--targetset cub --tau 0.1 --n_unlab_neg 64 \
--name 'cssf_WIEval_1shot/ewn_lpan/testing/arch-resnet12d_cosineLC_Pretrain-Src/cub-tgt_mIN-src/augSrc-False_augTgt-True/neg-L16U64_ftEP-300_lr-0.0001_tau-0.1' \
--load-modelpath '/jet/home/rajshekd/projects/PRALIGN/CSSF/output2/miniImagenet_ResNet12_rfs/mini_distilled.pth' \
--unlab_split './filelists/miniImagenet/base.json' \
--outfile \
--run_id $1
# --load-modelpath '/jet/home/rajshekd/projects/fsl_ssl/checkpoints/miniImagenet/_ResNet10_protonet_aug_5way_5shot_15query_rotation_lbda0.50Adam_lr0.0010/best_model.tar' \

# # ~~~~~ ewn_lpan_mtce / lpan_mtce ~~~~~
# CUDA_VISIBLE_DEVICES=0  python finetune_ConCe_parallelized.py \
# --ft_mode 'ewn_lpan_mtce' \
# --is_tgt_aug --n_shot 1 \
# --train_aug --method 'weight_imprint' --model 'ResNet10' \
# --stop_epoch 200 --num_tasks 20 --lr 0.005 \
# --targetset cub --tau 0.1 --n_unlab_neg 64 --cosine_fac 1.0 \
# --name 'cssf_WIEval_1shot/ewn_lpan_mtce/testing/arch-ResNet10_cosineLC_Pretrain-Src/cub-tgt_mIN-src/augSrc-False_augTgt-True/neg-L16U64_srcbsz-s64_ftEP-200_lr-0.005_tau-0.1_cosfac-1.0' \
# --load-modelpath 'output2/miniImagenet_ResNet10_lft/399.tar' \
# --unlab_split './filelists/miniImagenet/base.json' \
# --outfile \
# --run_id $1
# # --projection-head

# ####################### BASELINES #####################################

# # ~~~~~ ce ~~~~~
# CUDA_VISIBLE_DEVICES=0  python finetune_ConCe_parallelized.py \
# --ft_mode 'ce' \
# --train_aug --method 'weight_imprint' --model 'ResNet10' \
# --stop_epoch 200 --num_tasks 100 --lr 0.0005 \
# --targetset places \
# --is_tgt_aug --n_shot 5 \
# --name 'cssf_WIEval_5shot/ce/testing/arch-ResNet10_cosineLC_Pretrain-Src/places-tgt_mIN-src/augSrc-False_augTgt-True/bsz-25_ftEP-200_lr-0.0005' \
# --load-modelpath 'output2/miniImagenet_ResNet10_lft/399.tar' \
# --unlab_split './filelists/miniImagenet/base.json' \
# --outfile \
# --run_id $1

# # ~~~~~ ce mtce ~~~~~
# CUDA_VISIBLE_DEVICES=0  python finetune_ConCe_parallelized.py \
# --train_aug --method 'weight_imprint' --model 'ResNet10' \
# --is_tgt_aug --n_shot 1 \
# --ft_mode 'ce_mtce' \
# --stop_epoch 200 --num_tasks 20  --lr 0.005 \
# --targetset places --n_unlab_neg 128 --cosine_fac 1.0 \
# --name 'cssf_WIEval_1shot/ce_mtce/testing/arch-ResNet10_cosineLC_Pretrain-Src/places-tgt_mIN-src/augSrc-False_augTgt-True/bsz-s128t25_ftEP-200_lr-0.005_cosf-1.0' \
# --load-modelpath 'output2/miniImagenet_ResNet10_lft/399.tar' \
# --unlab_split './filelists/miniImagenet/base.json' \
# --outfile \
# --run_id $1


# ####################### ABLATIONS #####################################

# # ~~~~~ LPUN/LPAN ablation ~~~~~
# # 500 tasks for hyperparm selection
# CUDA_VISIBLE_DEVICES=0  python finetune_ConCe_parallelized.py \
# --ft_mode 'lpun' \
# --train_aug --method 'weight_imprint' --model 'ResNet10' \
# --stop_epoch 300 --num_tasks 20  --lr 0.005 \
# --targetset cars --tau 0.1 --n_unlab_neg 64 \
# --is_tgt_aug \
# --name 'cssf_WIEval/ConCE/lpun/testing/arch-ResNet10_featProj64_cosineLC_Pretrain-Src/cars-tgt_mIN-src/augSrc-False_augTgt-True/neg-L16U64_srcbsz-s64_ftEP-300_lr-0.005_tau-0.1' \
# --load-modelpath 'output2/miniImagenet_ResNet10_lft/399.tar' \
# --unlab_split './filelists/miniImagenet/base.json' \
# --outfile \
# --run_id $1 # command line argument for pid
# # --projection-head
# # --is_src_aug

# # ~~~~~ Source (Distractor) Subset ~~~~~
# pid=11
# CUDA_VISIBLE_DEVICES=0 python finetune_cssf-SrcSubset_parallelized.py \
# --train_aug --method 'weight_imprint' --model 'ResNet10' \
# --ft_mode 'LPAN' --stop_epoch 600 --num_tasks 50 --lr 0.001 \
# --targetset cub --tau 0.1 --n_unlab_neg 1024 \
# --name 'cssf_LinearEval/LPAN_SrcSubset/testing/arch-ResNet10_Pretrain-Src/cub_lab_mIN_unlab/distractorSz-1024/neg-L16U64_ftEP-600_lr-0.001_fastImpl' \
# --load-modelpath 'output2/miniImagenet_ResNet10_lft/399.tar' \
# --unlab_split './filelists/miniImagenet/base.json' \
# --outfile \
# --run_id $pid
# # --hyperparam_select \

# # ######################## VARIANTS #####################################

# # ~~~~~ tar_lpan (task aligned reweighting, higher the wt) ~~~~~
# CUDA_VISIBLE_DEVICES=0  python finetune_ConCe_parallelized.py \
# --ft_mode 'tar_lpan' --alpha 1.0 \
# --train_aug --method 'weight_imprint' --model 'ResNet10' \
# --stop_epoch 200 --num_tasks 71  --lr 0.005 \
# --targetset cars --tau 0.1 --n_unlab_neg 64 \
# --is_tgt_aug --n_shot 5 \
# --name 'cssf_WIEval_5shot/tar_lpan/testing/arch-ResNet10_cosineLC_Pretrain-Src/cars-tgt_mIN-src/augSrc-False_augTgt-True/neg-L16U64_ftEP-200_lr-0.005_tau-0.1_alpha-1.0' \
# --load-modelpath 'output2/miniImagenet_ResNet10_lft/399.tar' \
# --unlab_split './filelists/miniImagenet/base.json' \
# --outfile \
# --run_id $1
# #  --n_shot 1
# # --is_src_aug

# # ~~~~~ SemiSupcon/simclr ~~~~~
# # 500 tasks for hyperparm selection
# pid=0
# CUDA_VISIBLE_DEVICES=0  python finetune_SemiSupcon_parallelized.py \
# --ft_mode 'ewn_lpan' \
# --is_tgt_aug --n_shot 5 \
# --train_aug --method 'weight_imprint' --model 'ResNet10' \
# --stop_epoch 200 --num_tasks 20  --lr 0.005 \
# --targetset plantae --tau 0.1 --n_unlab_neg 128 \
# --name 'cssf_WIEval_5shot/SemiSupcon/ewn_lpan/testing/arch-ResNet10_cosineLC_Pretrain-Src/plantae-tgt_mIN-src/augSrc-False_augTgt-True/neg-L16U128_ftEP-200_lr-0.005_tau-0.1' \
# --load-modelpath 'output2/miniImagenet_ResNet10_lft/399.tar' \
# --unlab_split './filelists/miniImagenet/base.json' \
# --outfile \
# --run_id $1

# # ~~~~~ LPAN_CB (complete batch)/ LPAN_supcon/ ewn_lpan_cb ~~~~~
# # 500 tasks for hyperparm selection
# pid=0
# CUDA_VISIBLE_DEVICES=0  python finetune_ConCe_parallelized.py \
# --ft_mode 'ewn_lpan_cb' \
# --is_tgt_aug --n_shot 5 \
# --train_aug --method 'weight_imprint' --model 'ResNet10' \
# --stop_epoch 100 --num_tasks 20  --lr 0.005 \
# --targetset plantae --tau 0.1 --n_unlab_neg 128 \
# --name 'cssf_WIEval_5shot/ewn_lpan_cb/testing/arch-ResNet10_cosineLC_Pretrain-Src/plantae-tgt_mIN-src/augSrc-False_augTgt-True/neg-L16U128_ftEP-100_lr-0.005_tau-0.1' \
# --load-modelpath 'output2/miniImagenet_ResNet10_lft/399.tar' \
# --unlab_split './filelists/miniImagenet/base.json' \
# --outfile \
# --run_id $1
# # --projection-head
# # --is_src_aug

# # ~~~~~ APAN_CB (complete batch) ~~~~~
# # 500 tasks for hyperparm selection
# CUDA_VISIBLE_DEVICES=0  python finetune_ConCe_parallelized.py \
# --ft_mode 'apan_cb' \
# --is_tgt_aug --n_shot 1 \
# --train_aug --method 'weight_imprint' --model 'ResNet10' \
# --stop_epoch 200 --num_tasks 20  --lr 0.005 \
# --targetset plantae --tau 0.1 --n_unlab_neg 128 \
# --name 'cssf_WIEval_1shot/apan_cb/testing/arch-ResNet10_cosineLC_Pretrain-Src/plantae-tgt_mIN-src/augSrc-False_augTgt-True/neg-L16U128_ftEP-200_lr-0.005_tau-0.1' \
# --load-modelpath 'output2/miniImagenet_ResNet10_lft/399.tar' \
# --unlab_split './filelists/miniImagenet/base.json' \
# --outfile \
# --run_id $1
# # --projection-head
# # --is_src_aug

# # ~~~~~ LPAN Mixup ~~~~~
# pid=0
# CUDA_VISIBLE_DEVICES=0  python finetune_Mixup_parallelized.py --ft_mode 'lpan' \
# --train_aug --method 'weight_imprint' --model 'ResNet10' \
# --stop_epoch 100 --num_tasks 100  --lr 0.001 \
# --targetset places --tau 0.1 --n_unlab_neg 64 \
# --is_tgt_aug --alpha 0.8 \
# --name 'cssf_WIEval/Mixup/LPAN/testing/arch-ResNet10_cosineLC_Pretrain-Src/places-tgt_mIN-src/mixupSrc-augTgt/neg-L16U64_srcbsz-s64_ftEP-100_lr-0.001_alpha-0.8' \
# --load-modelpath 'output2/miniImagenet_ResNet10_lft/399.tar' \
# --unlab_split './filelists/miniImagenet/base.json' \
# --outfile \
# --run_id $1
# # --projection-head
# # --is_src_aug

# # ~~~~~ LPA1N HPM + INTERPOLATE ~~~~~
# RUNS=0
# for pid in `seq 0 ${RUNS}`
# do
# CUDA_VISIBLE_DEVICES=$(($pid)) python finetune_cssf-hpm-interp_parallelized.py \
# --train_aug --method 'weight_imprint' --model 'ResNet10' \
# --ft_mode 'LPAN' --stop_epoch 100 --num_tasks 600 \
# --targetset chestX --tau 0.1 --n_unlab_neg 64 --augstrength 'relaxed' \
# --alpha 0.9 \
# --name '' \
# --load-modelpath 'output2/miniImagenet_ResNet10_lft/399.tar' \
# --unlab_split './filelists/miniImagenet/base.json' \
# --run_id $1
# done

# # ~~~~~ conft_siamese ~~~~~
# pid=0
# CUDA_VISIBLE_DEVICES=0  python finetune_cssf-siamese_parallelized.py \
# --train_aug --method 'weight_imprint' --model 'ResNet10' \
# --stop_epoch 600 --num_tasks 50 --lr 0.001 \
# --targetset cub --tau 0.1 --n_unlab_neg 64 \
# --is_src_aug --is_tgt_aug \
# --name 'cssf_WIEval/LPAN-siamese/testing/arch-ResNet10_Pretrain-Src/cub_lab_mIN_unlab/augSrc-True_augTgt-True/neg-L16U64_ftEP-600_lr-0.001' \
# --load-modelpath 'output2/miniImagenet_ResNet10_lft/399.tar' \
# --unlab_split './filelists/miniImagenet/base.json' \
# --outfile \
# --run_id $1

