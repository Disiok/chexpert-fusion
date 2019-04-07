#!/bin/bash

cd /home/kelvin.wong/Developer/chexpert-fusion/fusion_experiments/
python func_run.py train_fusion_baselines \
    --shuffle \
    --cuda-benchmark \
    --num-gpus 1 \
    --num-epochs 10 \
    --num-workers 16 \
    --train-batch-size 12 \
    --val-batch-size 12 \
    --criterion focal_loss \
    --normalization groupnorm32 \
    --train-data /home/kelvin.wong/Datasets/CheXpert-v1.0 \
    --val-data /home/kelvin.wong/Datasets/CheXpert-v1.0 \
    --outdir /home/kelvin.wong/experiments/chexpert_fusion_baselines_early_1x12_gn32_relu_lr_0001_focal_loss_gamma_1/
