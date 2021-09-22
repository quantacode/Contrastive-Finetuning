#!/bin/bash

# Experimental Settings and Hyperparameters 
MODE='ewn_lpan'
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
--name Mode-${MODE}/TgtSet-${TgtSet}_DSET-${DSET}/DtracBsz-${DtracBsz}_FTepoch-${FTepoch}_TAU-${TAU} \
--load-modelpath 'output/checkpoints/baseline/399.tar' \
