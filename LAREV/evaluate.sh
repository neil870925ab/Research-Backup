#!/usr/bin/env bash

device=$1
split=$2 # test
model_name=$3 # t5-large
task=$4 # ECQA / ESNLI
use_irm=$5 # 1 or 0
irm_penalty_weight=$6 # lambda for imrv1
use_leak_probe=$7 # 1 or 0
leak_penalty_weight=$8 # leak probe penalty weight (e.g., 0.1, 0.3, 1.0)

python -m rev_eval \
        --task ${task} \
        --model_name ${model_name} \
        --split ${split} \
        --beams 2 \
        --device ${device} \
        --min_length 1 \
        --use_irm ${use_irm} \
        --irm_penalty_weight ${irm_penalty_weight} \
        --use_leak_probe ${use_leak_probe} \
        --leak_penalty_weight ${leak_penalty_weight} \
