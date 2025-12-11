#!/usr/bin/env bash
# env definf
device=$1
task=$2 # ECQA / ESNLI

OUT_DIR="./output"
MASK_TOKEN="<mask>"
MODEL_NAME="t5-large"
MODEL_DIR="./output/${task}_temp-${MODEL_NAME}"

python compute_ig.py \
	--device ${device} \
        --task ${task} \
        --model_dir "${MODEL_DIR}" \
        --device ${device} \
        --mask_token ${MASK_TOKEN}
