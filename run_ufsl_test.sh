#!/usr/bin/env bash


# ##################################################################
# ~~~~~ ewn_lpan (mIN) ~~~~~
RUNS=7
for pid in `seq 0 ${RUNS}`
do
CUDA_VISIBLE_DEVICES=$((($pid)))  python finetune_ConCe_parallelized.py \
--ft_mode 'ewn_lpan' \
--ufsl_dataset \
--is_tgt_aug --n_shot 5 --test_n_way 5 --n_query 15  \
--train_aug --method 'weight_imprint' --model 'Conv4_medium' \
--stop_epoch 400 --num_tasks 86 --lr 0.0005 \
--targetset miniImagenet --tau 0.05 --n_unlab_neg 64 \
--name 'cssf_WIEval_5shot/ewn_lpan/testing/arch-Conv4_medium_cosineLC_Pretrain-Src/miniImagenet-tgt_mIN-src/augSrc-False_augTgt-True/neg-L16U64_ftEP-400_lr-0.0005_tau-0.05' \
--load-modelpath '/home/rajshekd/projects/FSG/moco_v2/experiments/bsz-256/lr-0.03/snap.pth' \
--run_id $pid &
done

# # ~~~~~ ce (CelebA) ~~~~~
# RUNS=2
# for pid in `seq 0 ${RUNS}`
# do
# CUDA_VISIBLE_DEVICES=$((($pid)/3))  python finetune_ConCe_parallelized.py \
# --ft_mode 'ce' \
# --ufsl_dataset \
# --is_tgt_aug --n_shot 5 --test_n_way 2 --n_query 5  \
# --train_aug --method 'weight_imprint' --model 'Conv4_medium' \
# --stop_epoch 1000 --num_tasks 166 --lr 0.001 \
# --targetset celeba --cosine_fac 10.0 \
# --name 'cssf_WIEval_5shot/ce/testing/arch-Conv4_medium_cosineLC_Pretrain-Src/celeba-tgt_mIN-src/augSrc-False_augTgt-True/bsz-25_ftEP-1000_lr-0.001_cosfac-10.0' \
# --load-modelpath '/home/rajshekd/projects/FSG/FSG_raj/CSSF/output2/simclr/arch-Conv4_medium_Proj128_Pretrain-random/celeba_TARGETSET/nway-512_naug-2_tau-0.1_Aug3/chkpt/model_600.tar' \
# --run_id $pid &
# done
