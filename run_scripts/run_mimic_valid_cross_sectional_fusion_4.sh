#!/bin/bash

cd /home/suo/dev/chexpert-fusion/fusion_experiments
python func_run.py train_cross_sectional_fusion \
    --dataset-class PairedOnlyMIMICDataset \
    --label-class paper \
    --map-unobserved-to-negative \
    --evaluate-once \
    --use-test-set \
    --shuffle \
    --cuda-benchmark \
    --num-gpus 1 \
    --num-epochs 10 \
    --num-workers 16 \
    --train-batch-size 8 \
    --val-batch-size 8 \
    --fusion-index 4 \
    --train-data /home/suo/data/MIMIC-CXR \
    --val-data /home/suo/data/MIMIC-CXR \
    --outdir /home/suo/experiments/chexpert_cross_sectional_fusion_valid_mimic \
    --checkpoint /home/suo/experiments/chexpert_cross_sectional_fusion_unobserved_negative_4/models/model_best.pth.tar
