#!/bin/bash

cd /mnt/yyz_data_1/users/suo/dev/chexpert-fusion/fusion_experiments
python func_run.py train_volumetric_fusion \
    --shuffle \
    --cuda-benchmark \
    --num-gpus 1 \
    --num-epochs 10 \
    --num-workers 16 \
    --train-batch-size 4 \
    --val-batch-size 4 \
    --fusion-index 1 \
    --train-data /home/suo/data/CheXpert-v1.0 \
    --val-data /home/suo/data/CheXpert-v1.0 \
    --outdir /home/suo/experiments/chexpert_volumetric_fusion
