#!/bin/bash

cd /home/suo/dev/chexpert-fusion/fusion_experiments
python func_run.py train_cross_sectional_fusion \
    --dataset-class PairedOnlyCustomSplit \
    --label-class paper \
    --shuffle \
    --cuda-benchmark \
    --num-gpus 1 \
    --num-epochs 10 \
    --num-workers 16 \
    --train-batch-size 8 \
    --val-batch-size 8 \
    --fusion-index 1 \
    --train-data /home/suo/data/CheXpert-v1.0 \
    --val-data /home/suo/data/CheXpert-v1.0 \
    --outdir /home/suo/experiments/chexpert_cross_sectional_fusion_paired_only_custom_split
