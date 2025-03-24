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

import matplotlib.pyplot as plt

from .framework import GeSCF
from .utils import calculate_metric, show_mask_new


def test_single_image(dataset, img_t0_path, img_t1_path, gt_path=None, save_img=False):
    model = GeSCF(dataset=dataset, feature_facet='key', feature_layer=17, embedding_layer=32)
    
    # load image pairs
    img_t0 = cv2.imread(img_t0_path)
    rgb_img_t0 = cv2.cvtColor(img_t0, cv2.COLOR_BGR2RGB)
    img_t1 = cv2.imread(img_t1_path)
    rgb_img_t1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2RGB)
           
    # inference
    final_change_mask = model(img_t0_path, img_t1_path)    
    
    if gt_path:
        gt = cv2.imread(gt_path, 0) / 255.
        precision, recall = calculate_metric(gt, final_change_mask)
        f1score = 2 * (precision * recall) / (precision + recall)
    
    # visualization
    fig = plt.figure(figsize=(18,6))
    fig.add_subplot(141)
    plt.title('img t0')
    plt.imshow(rgb_img_t0)
    plt.axis('off')

    fig.add_subplot(142)
    plt.title('img t1')
    plt.imshow(rgb_img_t1)
    plt.axis('off')
    
    fig.add_subplot(143)
    plt.title('final change mask')
    plt.imshow(rgb_img_t0)
    show_mask_new(final_change_mask.astype(np.float32), plt.gca())
    plt.axis('off')
    
    if gt_path:
        fig.add_subplot(144)
        plt.title('GT')
        plt.imshow(rgb_img_t0)
        show_mask_new(gt.astype(np.float32), plt.gca())
        plt.axis('off')
    
    plt.show()
    
    #### 
    
    fig = plt.figure(figsize=(10,10))
    plt.imshow(rgb_img_t0)
    show_mask_new(final_change_mask.astype(np.float32), plt.gca())
    plt.axis('off')
    plt.savefig(f'./image/{dataset}.png',  bbox_inches='tight', pad_inches=0)
    plt.show()
    
    
    
    del model
    if gt_path:
        logging.info(f'Precision: {precision*100:.1f}, Recall: {recall*100:.1f}, F1: {f1score*100:.1f}')
        return precision, recall, f1score

if __name__ == '__main__':
    
    dataset = None
    split = None # one of SF-XL/St Lucia/Nordland

    img_t0_path = None
    img_t1_path = None
    gt_path = None

    test_single_image(dataset, img_t0_path, img_t1_path, gt_path)
    
