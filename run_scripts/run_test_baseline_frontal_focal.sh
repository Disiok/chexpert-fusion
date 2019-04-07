#!/bin/bash

cd /home/suo/dev/chexpert-fusion/fusion_experiments
python func_run.py train_baselines \
    --dataset-class PairedOnlyCustomSplit \
    --label-class paper \
    --view frontal \
    --map-unobserved-to-negative \
    --evaluate-once \
    --use-test-set \
    --shuffle \
    --learning-rate 1e-5 \
    --cuda-benchmark \
    --num-gpus 1 \
    --num-epochs 10 \
    --num-workers 16 \
    --train-batch-size 8 \
    --val-batch-size 8 \
    --train-data /home/suo/data/CheXpert-v1.0 \
    --val-data /home/suo/data/CheXpert-v1.0 \
    --outdir /home/suo/experiments/chexpert_baseline_frontal_focal_test \
    --checkpoint /home/suo/experiments/chexpert_baseline_frontal_unobserved_negative_focal_fix/models/model_best.pth.tar \
