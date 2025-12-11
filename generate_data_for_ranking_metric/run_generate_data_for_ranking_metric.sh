#!/usr/bin/env bash
task=$1 # ECQA / ESNLI

OUT_DIR="./output/${task}"


python generate_data_for_ranking_metric.py \
        --task ${task} \
        --out_dir ${OUT_DIR} \

