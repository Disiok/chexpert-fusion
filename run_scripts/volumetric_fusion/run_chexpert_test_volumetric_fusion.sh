#!/bin/bash

cd /home/suo/dev/chexpert-fusion/fusion_experiments
python func_run.py train_volumetric_fusion \
    --dataset-class PairedOnlyCustomSplit \
    --label-class paper \
    --map-unobserved-to-negative \
    --evaluate-once \
    --use-test-set \
    --shuffle \
    --cuda-benchmark \
    --num-gpus 1 \
    --num-epochs 10 \
    --num-workers 16 \
    --train-batch-size 6 \
    --val-batch-size 6 \
    --train-data /home/suo/data/CheXpert-v1.0 \
    --val-data /home/suo/data/CheXpert-v1.0 \
    --outdir /home/suo/experiments/chexpert_test/volumetric_fusion \
    --checkpoint /home/suo/experiments/chexpert_train/volumetric_fusion/models/model_best.pth.tar
