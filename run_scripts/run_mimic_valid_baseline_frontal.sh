#!/bin/bash

cd /home/suo/dev/chexpert-fusion/fusion_experiments
python func_run.py train_baselines \
    --dataset-class PairedOnlyMIMICDataset \
    --label-class paper \
    --view frontal \
    --map-unobserved-to-negative \
    --evaluate-once \
    --shuffle \
    --learning-rate 1e-5 \
    --cuda-benchmark \
    --num-gpus 1 \
    --num-epochs 10 \
    --num-workers 16 \
    --train-batch-size 8 \
    --val-batch-size 8 \
    --train-data /home/suo/data/MIMIC-CXR \
    --val-data /home/suo/data/MIMIC-CXR \
    --outdir /home/suo/experiments/chexpert_baseline_frontal_valid_mimic \
    --checkpoint /home/suo/experiments/chexpert_baseline_frontal_unobserved_negative_fix/models/model_best.pth.tar \
