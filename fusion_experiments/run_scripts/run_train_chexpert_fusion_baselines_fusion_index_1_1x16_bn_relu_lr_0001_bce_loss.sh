#!/bin/bash

cd /home/kelvin.wong/Developer/chexpert-fusion/fusion_experiments/
python func_run.py train_fusion_baselines \
    --shuffle \
    --cuda-benchmark \
    --num-gpus 1 \
    --num-epochs 10 \
    --num-workers 16 \
    --train-batch-size 8 \
    --val-batch-size 8 \
    --fusion-index 1 \
    --train-data /home/kelvin.wong/Datasets/CheXpert-v1.0 \
    --val-data /home/kelvin.wong/Datasets/CheXpert-v1.0 \
    --outdir /home/kelvin.wong/experiments/chexpert_fusion_baselines_fusion_index_1_1x16_bn_relu_lr_0001_bce_loss/
