#!/bin/bash

cd /home/kelvin.wong/Developer/chexpert-fusion/fusion_experiments
python func_run.py train_baselines \
    --dataset-class PairedOnlyCustomSplit \
    --label-class paper \
    --view lateral \
    --map-unobserved-to-negative \
    --evaluate-once \
    --shuffle \
    --cuda-benchmark \
    --num-gpus 1 \
    --num-epochs 10 \
    --num-workers 16 \
    --train-batch-size 8 \
    --val-batch-size 8 \
    --train-data /home/kelvin.wong/Datasets/CheXpert-v1.0 \
    --val-data /home/kelvin.wong/Datasets/CheXpert-v1.0 \
    --outdir /home/kelvin.wong/experiments/chexpert_test/baseline_lateral \
    --checkpoint /home/kelvin.wong/experiments/chexpert_train/baseline_lateral_focal_loss/models/model_best.pth.tar
