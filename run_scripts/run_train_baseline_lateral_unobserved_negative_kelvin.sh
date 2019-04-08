#!/bin/bash

cd /home/kelvin.wong/Developer/chexpert-fusion/fusion_experiments
python func_run.py train_baselines \
    --dataset-class PairedOnlyCustomSplit \
    --label-class paper \
    --view lateral \
    --map-unobserved-to-negative \
    --val-frequency 1000 \
    --learning-rate 1e-5 \
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
    --outdir /home/kelvin.wong/experiments/chexpert_baseline_lateral_unobserved_negative
