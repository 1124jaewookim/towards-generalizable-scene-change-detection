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

import matplotlib.pyplot as plt

from .framework import GeSCF
from .utils import calculate_metric


def test_full_dataset(dataset, split=None, save_img=False):
    model = GeSCF(dataset=dataset, feature_facet='key', feature_layer=17, embedding_layer=32)
    
    precisions = []
    recalls = []
    
    # example: test on VL-CMU-CD dataset
    if dataset == 'VL_CMU_CD':
        path_t0 = None
        path_t1 = None
        path_gt = None
        t0_images = os.listdir(path_t0)
        t1_images = os.listdir(path_t1)
        gt_images = os.listdir(path_gt)
          
    elif dataset == 'TSUNAMI':
        path_t0 = None
        path_t1 = None
        path_gt = None
        t0_images = os.listdir(path_t0)
        t1_images = os.listdir(path_t1)
        gt_images = os.listdir(path_gt)
        
    elif dataset == 'ChangeVPR':
        split = split
        path_t0 = None
        path_t1 = None
        path_gt = None
        t0_images = os.listdir(path_t0)
        t1_images = os.listdir(path_t1)
        gt_images = os.listdir(path_gt)
        
          
    pbar = tqdm(zip(t0_images, t1_images, gt_images), total=len(t0_images))
    for n, (t0, t1, gt) in enumerate(pbar):
        torch.cuda.empty_cache()
        t0_path = path_t0 + '/' + t0
        t1_path = path_t1 + '/' + t1
        gt_path = path_gt + '/' + gt
        
        gt = cv2.imread(gt_path, 0)   
        final_change_mask = model(t0_path, t1_path)
        
        prediction = final_change_mask
        precision, recall = calculate_metric(gt, prediction)
        
        precisions.append(precision)
        recalls.append(recall)
        
        current_precision = sum(precisions) / len(precisions)
        current_recall = sum(recalls) / len(recalls)
        current_f1score = 2 * (current_precision * current_recall) / (current_precision + current_recall)

        pbar.set_description(f"Processing {n+1}/{len(t0_images)}")
        pbar.set_postfix({
            "Precision": f"{current_precision:.4f}",
            "Recall": f"{current_recall:.4f}",
            "F1": f"{current_f1score:.4f}"
        })
        
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    f1score = 2 * (precision * recall) / (precision + recall)   
    del model  
    
    return precision, recall, f1score

if __name__ == '__main__':

    dataset = 'ChangeVPR'
    splits = ['SF-XL', 'Nordland', 'St Lucia']
    for split in splits:
        precision, recall, f1score = test_full_dataset(dataset, split=split)
        logging.info(f'Precision: {precision*100:.1f}, Recall: {recall*100:.1f}, F1: {f1score*100:.1f}')
    
    
    datasets = ['TSUNAMI', 'VL_CMU_CD']
    for dataset in datasets:
        precision, recall, f1score = test_full_dataset(dataset)
        logging.info(f'Precision: {precision*100:.1f}, Recall: {recall*100:.1f}, F1: {f1score*100:.1f}')
