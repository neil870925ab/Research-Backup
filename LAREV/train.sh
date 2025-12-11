#!/usr/bin/env bash
# env definf
device=$1
data_type=$2 # regular (r, b) / temp (b)
task=$3 # ECQA / COSE / ESNLI / QUARTZ (--logging_steps 100)
epochs=$4
lr=$5
use_leak_probe=$6 # 1 or 0
leak_penalty_weight=$7 # leak probe penalty weight (e.g., 0.1, 0.3, 1.0)

MODEL_NAME="t5-large"
PSI_MODEL_PATH="./output/${task}-${MODEL_NAME}_leak_probe_model/psi_model_weight.bin"

avoid_dot_in_filename=$(echo ${leak_penalty_weight} | sed 's/\./p/g')

if [ "$use_leak_probe" -eq 1 ]; then
    OUT_DIR="./output/${task}_${data_type}-${MODEL_NAME}-leak_probe_penalty_${avoid_dot_in_filename}"
else
    OUT_DIR="./output/${task}_${data_type}-${MODEL_NAME}"
fi

python rev_train.py \
        --task ${task} \
        --data_type ${data_type} \
        --out_dir ${OUT_DIR} \
        --model_name_or_path ${MODEL_NAME} \
        --device ${device} \
        --num_train_epochs ${epochs} \
        --learning_rate ${lr} \
        --do_train \
        --do_eval \
        --eval_during_train \
        --save_total_limit 1 \
        --overwrite_cache \
        --max_input_length 300 \
        --min_length 1 \
        --max_length 20 \
        --logging_steps 32 \
        --gradient_accumulation_steps 8 \
        --train_batch_size 8 \
        --eval_batch_size 8 \
        --overwrite_out_dir \
        --beams 2 \
        --use_leak_probe ${use_leak_probe} \
        --psi_model_path ${PSI_MODEL_PATH} \
        --leak_penalty_weight ${leak_penalty_weight}

