#!/bin/bash

cd /home/suo/dev/chexpert-fusion/fusion_experiments/
python func_run.py train_fusion_baselines \
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
    --train-batch-size 8 \
    --val-batch-size 8 \
    --fusion-index 4 \
    --train-data /home/suo/data/CheXpert-v1.0 \
    --val-data /home/suo/data/CheXpert-v1.0 \
    --outdir /home/suo/experiments/fusion_baselines_4_test \
    --checkpoint /home/suo/experiments/chexpert_fusion_baseline_fusion_index_4_1x8_bn_relu_lr_0001_bce_loss/models/model_best.pth.tar
