#!/bin/bash

cd /home/kelvin.wong/Developer/chexpert-fusion/fusion_experiments/
python func_run.py train_volumetric_fusion \
    --dataset-class PairedOnlyCustomSplit \
    --label-class paper \
    --map-unobserved-to-negative \
    --val-frequency 1000 \
    --shuffle \
    --cuda-benchmark \
    --num-gpus 1 \
    --num-epochs 10 \
    --num-workers 16 \
    --train-batch-size 6 \
    --val-batch-size 6 \
    --train-data /home/kelvin.wong/Datasets/CheXpert-v1.0 \
    --val-data /home/kelvin.wong/Datasets/CheXpert-v1.0 \
    --outdir /home/kelvin.wong/experiments/chexpert_train/volumetric_fusion/
