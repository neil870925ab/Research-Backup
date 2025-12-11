#!/usr/bin/env bash
# env definf
device=$1
task=$2 # ECQA / ESNLI
epochs=$3
lr=$4

OUT_DIR="./output"

python train_ig_model.py \
	--device ${device} \
        --task ${task} \
        --out_dir "${OUT_DIR}/${task}_ig_model" \
        --device ${device} \
        --epochs ${epochs} \
        --learning_rate ${lr} \
        --max_length 256 \
        --train_batch_size 16 \
