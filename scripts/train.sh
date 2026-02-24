#!/bin/bash

# Available datasets: APTOS, MESSIDOR, DEEPDR, DDR
DATASET="APTOS"
# Available backbones: vit_base_patch16_224, retfound_dinov2_meh
BACKBONE="vit_base_patch16_224"

# I/O Paths
DATASET_ROOT="/path/to/GDRBench/images"
SPLITS_ROOT="/path/to/GDRBench/splits"
SAVE_DIR="/path/to/save_dir"
LOG_DIR="/path/to/log_dir"

# Training parameters
BATCH_SIZE=128
EPOCHS=1
LR=1e-4
DTYPE="fp16"

SEED=42

CUDA_VISIBLE_DEVICES=0 python3 scripts/train.py --dataset "$DATASET" --dataset_root "$DATASET_ROOT" --splits_root "$SPLITS_ROOT" --batch_size $BATCH_SIZE --epochs $EPOCHS --lr $LR --dtype $DTYPE --save_dir "$SAVE_DIR" --log_dir "$LOG_DIR" --backbone "$BACKBONE" --seed $SEED
