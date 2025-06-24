"""
test on standard SCD datasets and ChangeVPR (or own image pairs)
"""
import os 
import numpy as np
import cv2
from tqdm import tqdm

import logging
logging.basicConfig(
    level=logging.INFO,               
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

import argparse
import matplotlib.pyplot as plt

from framework import GeSCF
from utils import calculate_metric, visualize_results


def evaluate_single_image_pair(args):
    model = GeSCF(args)

    # Load images
    img_t0 = cv2.imread(args.img_t0_path)
    img_t1 = cv2.imread(args.img_t1_path)
    rgb_img_t0 = cv2.cvtColor(img_t0, cv2.COLOR_BGR2RGB)
    rgb_img_t1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2RGB)

    # Run inference
    final_change_mask = model(args.img_t0_path, args.img_t1_path)

    # Evaluate metrics if GT is available
    if args.gt_path:
        gt = cv2.imread(args.gt_path, 0) / 255.
        precision, recall = calculate_metric(gt, final_change_mask)
        f1score = 2 * (precision * recall) / (precision + recall + 1e-9)
    else:
        gt, precision, recall, f1score = None, None, None, None

    # Visualize results
    visualize_results(rgb_img_t0, rgb_img_t1, final_change_mask, gt)

    del model

    # Log evaluation results
    if args.gt_path:
        logging.info(f'Precision: {precision*100:.1f}, Recall: {recall*100:.1f}, F1: {f1score*100:.1f}')
        return precision, recall, f1score


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Generalizable Scene Change Detection Framework (GeSCF)', add_help=add_help)
    
    ### Dataset
    parser.add_argument('--test-dataset', default='ChangeVPR', help='dataset name')
    parser.add_argument('--output-size', default=512, type=int, metavar='N', help='the input-size of images')
    parser.add_argument('--img-t0-path', default='', help='dataset name')
    parser.add_argument('--img-t1-path', default='', help='dataset name')
    parser.add_argument('--gt-path', default=None, help='dataset name')
    
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
    evaluate_single_image_pair(args)
    
