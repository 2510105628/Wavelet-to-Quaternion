#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
MODEL=mamba_vision_T
DATA_PATH="/home/ac/data/wfz/Imagenet1000"
BS=256
EXP=my_experiment
LR=5e-4
WD=0.05
DR=0.2
RESUME='/home/wfz/lxh/Mambavision/output/train/my_experiment/20250802-194945-mamba_vision_T-224/last.pth.tar'
MASTER_PORT=39501
torchrun --nproc_per_node=1  train.py --input-size 3 224 224 --crop-pct=0.875 \
--data_dir $DATA_PATH --train-split train --val-split val --model $MODEL --master-port ${MASTER_PORT} --resume $RESUME --amp --weight-decay ${WD} --drop-path ${DR} --batch-size $BS --tag $EXP --lr $LR