#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python test_single.py \
    --test-dataset Random \
    --output-size 512 \
    \
    --img-t0-path F:/GeSCD/ChangeVPR/SF-XL/t0/00000001.png \
    --img-t1-path F:/GeSCD/ChangeVPR/SF-XL/t1/00000001.png \
    --gt-path F:/GeSCD/ChangeVPR/SF-XL/mask/00000001.png \
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