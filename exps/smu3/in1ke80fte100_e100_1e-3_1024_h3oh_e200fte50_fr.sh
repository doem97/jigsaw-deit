#!/bin/bash

# h3 on jigsaw head, with freeze
# use stable learning rate 1e-3
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# export THREADS=4
# export WANDB_MODE=disabled

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export THREADS=4
export CONFIG_ID="in1ke80fte100_e100_1e-3_1024_h3oh_e200fte50_fr"

torchrun --nproc_per_node=$THREADS \
    --master_port 15000 \
    main_jigsaw.py \
    --model jigsaw_base_patch56_336 \
    --input-size 336 \
    --batch-size 256 \
    --epochs 50 \
    --unscale-lr \
    --lr 1e-3 \
    --min-lr 1e-6 \
    --sched cosine \
    --mask-ratio 0.0 \
    --bce-loss \
    --data-path "./data/food101/" \
    --data-set IMNET \
    --finetune "./outputs/jigsaw_base_p56_336_f101_shuffle_in1ke10fte300/best_checkpoint_e200.pth" \
    --use-cls \
    --freeze \
    --output_dir ./outputs/${CONFIG_ID} \
    --log_dir ./logs/${CONFIG_ID}