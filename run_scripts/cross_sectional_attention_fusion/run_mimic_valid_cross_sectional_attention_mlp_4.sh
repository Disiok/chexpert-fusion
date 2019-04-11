#!/bin/bash

cd /home/suo/dev/chexpert-fusion/fusion_experiments
python func_run.py train_cross_sectional_attention_fusion \
    --dataset-class PairedOnlyMIMICDataset \
    --label-class paper \
    --map-unobserved-to-negative \
    --fusion-index 4 \
    --fusion-operator cross_sectional_attention_mlp \
    --evaluate-once \
    --shuffle \
    --cuda-benchmark \
    --num-gpus 1 \
    --num-epochs 10 \
    --num-workers 16 \
    --train-batch-size 8 \
    --val-batch-size 8 \
    --train-data /home/suo/data/MIMIC-CXR \
    --val-data /home/suo/data/MIMIC-CXR \
    --outdir /home/suo/experiments/mimic_valid/cross_sectional_attention_fusion_mlp_0/ \
    --checkpoint /home/suo/experiments/chexpert_train/cross_sectional_attention_fusion_mlp_0/models/model_best.pth.tar
