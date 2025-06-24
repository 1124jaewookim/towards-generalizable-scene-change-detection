"""
test on standard SCD datasets and ChangeVPR (or own image pairs)
"""
import os 
import cv2
import numpy as np
from tqdm import tqdm

import torch
import logging
logging.basicConfig(
    level=logging.INFO,               
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

import argparse
import matplotlib.pyplot as plt

from framework import GeSCF
from utils import calculate_metric


def evaluate_dataset(args):
    model = GeSCF(args)

    precisions = []
    recalls = []

    path_t0 = os.path.join(args.dataset_path, "t0")
    path_t1 = os.path.join(args.dataset_path, "t1")
    path_gt = os.path.join(args.dataset_path, "mask")

    t0_images = sorted(os.listdir(path_t0))
    t1_images = sorted(os.listdir(path_t1))
    gt_images = sorted(os.listdir(path_gt))

    log_name = (
        f"{args.test_dataset}"
        f"_feat-{args.feature_facet}"
        f"_fl{args.feature_layer}"
        f"_el{args.embedding_layer}"
        f"_sam-{args.sam_backbone}"
        f"_psd-{args.pseudo_backbone}"
        f"_sz{args.output_size}.txt"
    )

    os.makedirs("./results", exist_ok=True)
    log_path = os.path.join("./results", log_name)
    
    pbar = tqdm(zip(t0_images, t1_images, gt_images), total=len(t0_images))
    for n, (t0, t1, gt) in enumerate(pbar):
        torch.cuda.empty_cache()

        t0_path = os.path.join(path_t0, t0)
        t1_path = os.path.join(path_t1, t1)
        gt_path = os.path.join(path_gt, gt)

        gt_mask = cv2.imread(gt_path, 0)
        prediction = model(t0_path, t1_path)

        precision, recall = calculate_metric(gt_mask, prediction)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

        precisions.append(precision)
        recalls.append(recall)

        # Running averages
        avg_precision = sum(precisions) / len(precisions)
        avg_recall = sum(recalls) / len(recalls)
        avg_f1score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall + 1e-9)

        # Log per-image F1 score
        filename = os.path.basename(t0_path)
        log_line = f"[{filename}] F1-score: {f1:.4f} [Running Avg F1: {avg_f1score:.4f}]"
        with open(log_path, "a") as f:
            f.write(log_line + "\n")

        # Progress bar update
        pbar.set_description(f"Processing {n + 1}/{len(t0_images)}")
        pbar.set_postfix({
            "Precision": f"{avg_precision:.4f}",
            "Recall": f"{avg_recall:.4f}",
            "F1": f"{avg_f1score:.4f}"
        })

    del model

    final_precision = sum(precisions) / len(precisions)
    final_recall = sum(recalls) / len(recalls)
    final_f1score = 2 * (final_precision * final_recall) / (final_precision + final_recall + 1e-9)

    return final_precision, final_recall, final_f1score


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Generalizable Scene Change Detection Framework (GeSCF)', add_help=add_help)
    
    ### Dataset
    parser.add_argument('--test-dataset', default='ChangeVPR', help='dataset name')
    parser.add_argument('--output-size', default=512, type=int, metavar='N', help='the input-size of images')
    parser.add_argument('--dataset-path', default='F:/GeSCD/ChangeVPR/SF-XL', help='total dataset path')
    
    ### Model
    parser.add_argument('--feature-facet', default='key', help='feature-facet to intercept')
    parser.add_argument('--feature-layer', default=17, type=int, help='ViT layer to intercept featire-facet')
    parser.add_argument('--embedding-layer', default=32, type=int, help='ViT layer to intercept image-embedding & mask-embedding')

    # SAM Backbone
    parser.add_argument('--sam-backbone', default='vit_h', help='backbone of sam automatic mask generator')
    parser.add_argument('--points-per-side', default=32, type=int, help='grid point density')
    parser.add_argument('--pred-iou-thresh', default=0.7, type=float, help='lower to get more mask')
    parser.add_argument('--stability-score-thresh', default=0.7, type=float, help='lower to get more mask')
    
    # PseudoGenerator Backbone
    parser.add_argument('--pseudo-backbone', default='vit_h', help='backbone of initial pseudo mask generator')
    
    return parser

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    precision, recall, f1score = evaluate_dataset(args)
    logging.info(f'Precision: {precision*100:.1f}, Recall: {recall*100:.1f}, F1: {f1score*100:.1f}')
