#!/bin/bash

export CUDA_VISIBLE_DEVICES="4,5,6,7"
export THREADS=4

python -m torch.distributed.launch \
    --nproc_per_node=$THREADS \
    --use_env \
    --master_port 30000 \
    main_jigsaw.py \
    --rec \
    --model jigsaw_r_base_p56_336 \
    --input-size 336 \
    --batch-size 512 \
    --num_workers 10 \
    --epochs 10 \
    --sched cosine \
    --unscale-lr \
    --lr 1e-3 \
    --min-lr 1e-6 \
    --mask-ratio 0.0 \
    --data-path /workspace/study/imagenet/ILSVRC/Data/CLS-LOC \
    --data-set IMNET \
    --lambda-rec 0.1 \
    --output_dir ./outputs/jigsaw_r_b_p56_336_in1k_m0_10e

    # --resume "./outputs/jigsaw_base_p56_336_in1k/checkpoint_2.pth" \
    # --finetune /workspace/study/jigsawvit/jigdeit/data/jigsaw_base_results/best_checkpoint.pth \
