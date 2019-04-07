#!/bin/bash

cd /home/suo/dev/chexpert-fusion/fusion_experiments
python func_run.py train_volumetric_fusion \
    --dataset-class PairedOnlyCustomSplit \
    --label-class paper \
    --map-unobserved-to-negative \
    --use-2d-conv \
    --val-frequency 1000 \
    --shuffle \
    --cuda-benchmark \
    --num-gpus 1 \
    --num-epochs 10 \
    --num-workers 16 \
    --train-batch-size 6 \
    --val-batch-size 6 \
    --train-data /home/suo/data/CheXpert-v1.0 \
    --val-data /home/suo/data/CheXpert-v1.0 \
    --outdir /home/suo/experiments/chexpert_volumetric_fusion_unobserved_negative_2d
