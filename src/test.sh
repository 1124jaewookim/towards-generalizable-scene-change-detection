#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python test.py \
    --test-dataset VL_CMU_CD \
    --output-size 512 \
    \
    --dataset-path F:/GeSCD/VL_CMU_CD/test \
    \
    --feature-facet key \
    --feature-layer 17 \
    --embedding-layer 32 \
    \
    --sam-backbone vit_h \
    --pseudo-backbone vit_h \
    \
    --points-per-side 32 \
    --pred-iou-thresh 0.7 \
    --stability-score-thresh 0.7