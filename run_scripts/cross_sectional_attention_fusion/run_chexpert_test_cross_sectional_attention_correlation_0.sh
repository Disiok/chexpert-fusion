#!/bin/bash

cd /home/kelvin.wong/Developer/chexpert-fusion/fusion_experiments
python func_run.py train_cross_sectional_attention_fusion \
    --dataset-class PairedOnlyCustomSplit \
    --label-class paper \
    --map-unobserved-to-negative \
    --evaluate-once \
    --use-test-set \
    --fusion-index 0 \
    --fusion-operator cross_sectional_attention_correlation \
    --val-frequency 1000 \
    --shuffle \
    --cuda-benchmark \
    --num-gpus 1 \
    --num-epochs 10 \
    --num-workers 16 \
    --train-batch-size 8 \
    --val-batch-size 8 \
    --train-data /home/kelvin.wong/Datasets/CheXpert-v1.0 \
    --val-data /home/kelvin.wong/Datasets/CheXpert-v1.0 \
    --outdir /home/kelvin.wong/experiments/chexpert_test/cross_sectional_attention_correlation_0/ \
    --checkpoint /home/kelvin.wong/experiments/chexpert_train/cross_sectional_attention_correlation_0/models/model_best.pth.tar
