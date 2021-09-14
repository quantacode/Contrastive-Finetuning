#!/bin/bash

# CONFT
MODE='ewn_lpan'
TgtSet=cub
DtracSet='./filelists/miniImagenet/base.json'
DtracBsz=128
FTepoch=100
TAU=0.05
# -------- Run command ---------
CUDA_VISIBLE_DEVICES=1  python finetune_ConCe_parallelized.py \
--ft_mode $MODE \
--targetset $TgtSet --is_tgt_aug \
--distractor_set $DtracSet \
--distractor_bsz $DtracBsz \
--stop_epoch $FTepoch --tau $TAU \
--name Mode-$MODE/TgtSet-$TgtSet_DSET-$DSET/DtracBsz-$DtracBsz_FTepoch-$FTepoch_TAU-$TAU \
--load-modelpath '/home/rajshekd/projects/FSG/FSG_raj/CSSF/output2/miniImagenet_ResNet10_lft/399.tar' \