#!/usr/bin/env bash

SPLIT='ranking'
RANKING_FILE='test_data_for_ranking_metric.jsonl'

device=$1
ranking_type=$2 # gold, gold_leaky, vacuous, leaky, truncated_gold_80, truncated_gold_50, gold_noise, shuffled_gold
model_name=$3 # t5-large
task=$4 # ECQA / ESNLI 
use_irm=$5 # 1 or 0
irm_penalty_weight=$6 # lambda for imrv1 (e.g., 0.2, 0.5, 1)
use_leak_probe=$7 # 1 or 0
leak_penalty_weight=$8 # leak probe penalty weight (e.g., 0.001, 0.005, 0.01)

python -m rev_eval \
        --task ${task} \
        --model_name ${model_name} \
        --split ${SPLIT} \
        --beams 2 \
        --device ${device} \
        --min_length 1 \
        --use_irm ${use_irm} \
        --irm_penalty_weight ${irm_penalty_weight} \
        --ranking_file ${RANKING_FILE} \
        --ranking_type ${ranking_type} \
        --use_leak_probe ${use_leak_probe} \
        --leak_penalty_weight ${leak_penalty_weight} \
