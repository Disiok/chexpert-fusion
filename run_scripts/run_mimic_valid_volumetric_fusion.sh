#!/bin/bash

cd /home/suo/dev/chexpert-fusion/fusion_experiments
python func_run.py train_volumetric_fusion \
    --dataset-class PairedOnlyMIMICDataset \
    --label-class paper \
    --map-unobserved-to-negative \
    --evaluate-once \
    --shuffle \
    --cuda-benchmark \
    --num-gpus 1 \
    --num-epochs 10 \
    --num-workers 16 \
    --train-batch-size 6 \
    --val-batch-size 6 \
    --train-data /home/suo/data/MIMIC-CXR \
    --val-data /home/suo/data/MIMIC-CXR \
    --outdir /home/suo/experiments/chexpert_volumetric_fusion_valid_mimic \
    --checkpoint /home/suo/experiments/chexpert_volumetric_fusion_unobserved_negative/models/model_best.pth.tar