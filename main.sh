#!/bin/bash

# Experimental Settings and Hyperparameters 
MODE='ewn_lpan' #options: [ewn_lpan/ewn_lpan_mtce], ewn_lpan=conft, ewn_lpan_mtce=conft_mtce
TgtSet=cub #options: [cub/cars/places/plantae]
DtracSet='./filelists/miniImagenet/base.json'
DtracBsz=128
FTepoch=100
TAU=0.05
# -------- Run command ---------
CUDA_VISIBLE_DEVICES=0  python finetune.py \
--ft_mode $MODE \
--targetset $TgtSet --is_tgt_aug \
--distractor_set $DtracSet \
--distractor_bsz $DtracBsz \
--stop_epoch $FTepoch --tau $TAU \
--name Mode-$MODE/TgtSet-$TgtSet_DSET-$DSET/DtracBsz-$DtracBsz_FTepoch-$FTepoch_TAU-$TAU \
--load-modelpath 'output/checkpoints/baseline/399.tar' \
