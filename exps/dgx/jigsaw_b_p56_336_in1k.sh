#!/bin/bash

export CUDA_VISIBLE_DEVICES="4,5,6,7"
export THREADS=4

python -m torch.distributed.launch \
    --nproc_per_node=$THREADS \
    --use_env \
    main_jigsaw.py \
    --model jigsaw_base_p56_336 \
    --input-size 336 \
    --batch-size 512 \
    --epochs 50 \
    --sched cosine \
    --unscale_lr \
    --lr 1e-3 \
    --unscale_lr \
    --min-lr 1e-6 \
    --warmup-epochs 1 \
    --mask-ratio 0.0 \
    --data-path /workspace/study/imagenet/ILSVRC/Data/CLS-LOC \
    --data-set IMNET \
    --output_dir ./outputs/jigsaw_base_p56_336_in1k
    # --finetune /workspace/study/jigsawvit/jigdeit/data/jigsaw_base_results/best_checkpoint.pth \