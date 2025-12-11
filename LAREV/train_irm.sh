#!/usr/bin/env bash
# Training script for IRMv1 domain generalization
# Usage: ./train_irm.sh <device> <task> <epochs> <lr> <penalty_weight> [anneal_iters]

device=$1
data_type=$2 # regular (r, b) only
task=$3  # ECQA / ESNLI
epochs=$4
lr=$5
irm_penalty_weight=$6  # IRMv1 penalty weight (e.g., 1, 10, 100)
irm_anneal_iters=$7
use_leak_probe=$8 # 1 or 0
leak_penalty_weight=$9 # leak probe penalty weight (e.g., 0.1, 0.3, 1.0)

MODEL_NAME="t5-large"
PSI_MODEL_PATH="./output/${task}-${MODEL_NAME}_leak_probe_model/psi_model_weight.bin"

avoid_dot_in_filename_lp=$(echo ${leak_penalty_weight} | sed 's/\./p/g')
avoid_dot_in_filename_irm=$(echo ${irm_penalty_weight} | sed 's/\./p/g')

if [ "$use_leak_probe" -eq 1 ]; then
    OUT_DIR="./output/${task}_${data_type}_irm-${MODEL_NAME}-penalty_${avoid_dot_in_filename_irm}-leak_probe_penalty_${avoid_dot_in_filename_lp}"
else
    OUT_DIR="./output/${task}_${data_type}_irm-${MODEL_NAME}-penalty_${avoid_dot_in_filename_irm}"
fi

python rev_train.py \
        --task ${task} \
        --data_type ${data_type} \
        --use_irm 1 \
        --irm_penalty_weight ${irm_penalty_weight} \
        --irm_penalty_anneal_iters ${irm_anneal_iters} \
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
        --logging_steps 512 \
        --gradient_accumulation_steps 1 \
        --train_batch_size 3 \
        --eval_batch_size 3 \
        --overwrite_out_dir \
        --beams 1 \
        --use_leak_probe ${use_leak_probe} \
        --psi_model_path ${PSI_MODEL_PATH} \
        --leak_penalty_weight ${leak_penalty_weight}
