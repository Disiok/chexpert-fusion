#!/bin/bash

cd /home/mbai3/csc2541/project/chexpert-fusion/fusion_experiments
python func_run.py train_cross_sectional_attention_fusion \
    --dataset-class PairedOnlyCustomSplit \
    --label-class paper \
    --val-frequency 1000 \
    --shuffle \
    --cuda-benchmark \
    --num-gpus 1 \
    --num-epochs 10 \
    --num-workers 16 \
    --train-batch-size 8 \
    --val-batch-size 8 \
    --fusion-index 1 \
    --train-data /media/mbai3/a84c7eb1-9838-4712-954f-2e0339592f0e/CheXpert-v1.0 \
    --val-data /media/mbai3/a84c7eb1-9838-4712-954f-2e0339592f0e/CheXpert-v1.0 \
    --outdir /media/mbai3/a84c7eb1-9838-4712-954f-2e0339592f0e/CheXpert-v1.0/experiments/chexpert_cross_sectional_attention_fusion_paired_only_custom_split
