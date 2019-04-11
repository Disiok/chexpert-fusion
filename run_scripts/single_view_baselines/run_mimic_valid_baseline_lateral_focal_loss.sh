#!/bin/bash

cd /home/suo/dev/chexpert-fusion/fusion_experiments
python func_run.py train_baselines \
    --dataset-class PairedOnlyMIMICDataset \
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
    --train-data /home/suo/data/MIMIC-CXR \
    --val-data /home/suo/data/MIMIC-CXR \
    --outdir /home/suo/experiments/mimic_valid/baseline_lateral \
    --checkpoint /home/suo/experiments/chexpert_train/baseline_lateral_focal_loss/models/model_best.pth.tar \
