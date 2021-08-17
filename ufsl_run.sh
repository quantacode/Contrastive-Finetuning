#!/usr/bin/env bash

# INDEX
# OL - only labelled contrastive loss
# CE - standard CE classfier
# LPAN - labelled positives and all negative contrastive loss

# # ~~~~~ finetune Only Labelled Contrastive (OL) ~~~~~
# RUNS=3
# for pid in `seq 0 ${RUNS}`
# do
# CUDA_VISIBLE_DEVICES=$(($pid+4)) python finetune_cssf_parallelized.py \
# --train_aug --method 'weight_imprint' --model 'ResNet10' \
# --ft_mode 'OL' --stop_epoch 100 --num_tasks 150 \
# --targetset CUB --tau 0.1 --augstrength 'relaxed' \
# --name 'cssf_LinearEval/OL/arch-ResNet10_Pretrain-Src/CUB_lab/neg-L16_ftEP-100_clsEP-50_tau-0.1' \
# --load-modelpath '/home/rajshekd/projects/FSG/FSG_raj/CSSF/output2/miniImagenet_ResNet10_lft/399.tar' \
# --run_id $pid &
# done
# # --load-modelpath '/home/rajshekd/projects/FSG/FSG_raj/LFT_cdfsl/output2/src-baseline_tgt-npUmtra/arch-ResNet10_Proj-linear_Pretrain-baseline/CUB_TARGETSET/nway-100_naug-2_losswt-1.0_aggressiveAug/SIMCLR/chkpt/best_model_tgtval.tar' \
# # --load-modelpath '/home/rajshekd/projects/FSG/FSG_raj/BS_cdfsl/output2/src-baseline_tgt-mvcon/arch-ResNet10_Proj-linear_Pretrain-baseline/eurosat_TARGETSET/baselineFeat-bkbn_weighting-uniform/nway-50_naug-2/chkpt/best_model_projhead.tar'
# # --load-modelpath '/home/rajshekd/projects/FSG/FSG_raj/BS_cdfsl/output2/src-baseline_tgt-mvcon/arch-ResNet10_Proj-linear_Pretrain-baseline/isic-practice_TARGETSET/baselineFeat-bkbn_weighting-uniform/nway-50_naug-1/chkpt/best_model_projhead.tar'
# # --load-modelpath '/home/rajshekd/projects/FSG/FSG_raj/CSSF/output2/miniImagenet_ResNet10_lft/399.tar'
# # --load-modelpath '/home/rajshekd/projects/FSG/FSG_raj/LFT_cdfsl/output2/npcon/ABLATION/simclr/arch-Conv4_Proj-linear_Pretrain-random/targetset-mIN_relaxedAug/nway-100_naug-2_tau-0.1/chkpt/best_model_projhead.tar'
# # --load-modelpath '/home/rajshekd/projects/FSG/FSG_raj/LFT_cdfsl/output2/npcon/ABLATION/simclr/arch-ResNet10_Proj-linear_Pretrain-tgtCon/targetset-CUB_relaxedAug/nway-20_naug-2_tau-0.1/chkpt/best_model_novel.tar'


# # ~~~~~ finetune Labelled Pos and All Neg Contrastive (LPAN) ~~~~~
# RUNS=4
# for pid in `seq 0 ${RUNS}`
# do
# CUDA_VISIBLE_DEVICES=$((($pid)/2)) python finetune_cssf_parallelized.py \
# --train_aug --method 'weight_imprint' --model 'Conv4_medium' \
# --ft_mode 'LPAN' --stop_epoch 300 --num_tasks 120 --lr 0.0005 \
# --ufsl_dataset --targetset miniImagenet \
# --tau 0.5 --n_unlab_neg 64 \
# --is_tgt_aug \
# --name 'cssf_WIEval/LPAN/testing/arch-Conv4_medium_Pretrain-tgtbase-LossModel/miniImagenet/augSrc-False_augTgt-True/neg-L16U64_ftEP-300_lr-0.0005_tau-0.5' \
# --load-modelpath '/home/rajshekd/projects/FSG/FSG_raj/LFT_cdfsl/output2/simclr/arch-Conv4_medium_Proj128_Pretrain-random/miniImagenet_TARGETSET/nway-512_naug-2_tau-0.1/chkpt/best_model_projhead.tar' \
# --run_id $pid &
# done
# # --n_shot 1
# # --is_src_aug
# # --hyperparam_select \


# ~~~~~ finetune CE ~~~~~
RUNS=5
for pid in `seq 0 ${RUNS}`
do
CUDA_VISIBLE_DEVICES=1  python finetune_cssf_parallelized.py \
--train_aug --method 'weight_imprint' --model 'Conv4_medium' \
--ft_mode 'CE' --stop_epoch 600 --num_tasks 100 --lr 0.0005 \
--ufsl_dataset --targetset miniImagenet \
--is_tgt_aug \
--name 'cssf_WIEval/CE/testing/arch-Conv4_medium_cosineLC_Pretrain-tgtbase/miniImagenet_lab/augSrc-False_augTgt-True/bsz-L25_ftEP-600_lr-0.0005' \
--load-modelpath '/home/rajshekd/projects/FSG/FSG_raj/LFT_cdfsl/output2/simclr/arch-Conv4_medium_Proj128_Pretrain-random/miniImagenet_TARGETSET/nway-512_naug-2_tau-0.1/chkpt/best_model_projhead.tar' \
--run_id $pid &
done
# --n_shot 1 \
# --hyperparam_select \

