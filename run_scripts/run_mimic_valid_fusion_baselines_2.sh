#!/bin/bash

cd /home/suo/dev/chexpert-fusion/fusion_experiments/
python func_run.py train_fusion_baselines \
    --dataset-class PairedOnlyMIMICDataset \
    --label-class paper \
    --map-unobserved-to-negative \
    --evaluate-once \
    --shuffle \
    --cuda-benchmark \
    --num-gpus 1 \
    --num-epochs 10 \
    --num-workers 16 \
    --train-batch-size 8 \
    --val-batch-size 8 \
    --fusion-index 2 \
    --train-data /home/suo/data/MIMIC-CXR \
    --val-data /home/suo/data/MIMIC-CXR \
    --outdir /home/suo/experiments/fusion_baselines_2_valid_mimic \
    --checkpoint /home/suo/experiments/chexpert_fusion_baseline_fusion_index_2_1x8_bn_relu_lr_0001_bce_loss/models/model_best.pth.tar
