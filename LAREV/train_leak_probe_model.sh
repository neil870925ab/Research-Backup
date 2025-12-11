#!/usr/bin/env bash
# env definf
device=$1
task=$2 # ECQA / ESNLI
epochs=$3
lr=$4

MODEL_NAME="t5-large"
OUT_DIR="./output"
OUT_NAME="psi_model_weight.bin"

python train_leak_probe_model_psi.py \
        --task ${task} \
        --out_dir "${OUT_DIR}/${task}-${MODEL_NAME}_leak_probe_model" \
        --out_name ${OUT_NAME} \
        --model_name_or_path ${MODEL_NAME} \
        --device ${device} \
        --num_train_epochs ${epochs} \
        --learning_rate ${lr} \
        --max_input_length 300 \
        --max_length 20 \
        --train_batch_size 16 \
        --eval_batch_size 16 \
        --phi_model_path "${OUT_DIR}/${task}_regular-${MODEL_NAME}" \
