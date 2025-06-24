"""
Generalizable Scene Change Detection Framework (GeSCF)
"""

import logging
import cv2
import numpy as np
from scipy.stats import skew

import torch
import torch.nn as nn
import torchvision

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Internal utilities
from utils import calculate_iou, sanity_check_args, load_backbones

# Modules
from segment_anything_model import SamAutomaticMaskGenerator
from pseudo_generator import PseudoGenerator
from registration import coarse_transform


class GeSCF(nn.Module):
    def __init__(self, args):
        sanity_check_args(args)
        super(GeSCF, self).__init__()

        # Dataset settings
        self.dataset = args.test_dataset
        self.dataset_bias = self.dataset == 'VL_CMU_CD'
        
        self.img_size = (256, 256) if self.dataset == 'TSUNAMI' else (512, 512)
        self.output_size = (args.output_size, args.output_size)
        
        logging.info(f'dataset name: {self.dataset}')

        # Feature extraction settings
        self.feature_facet = args.feature_facet
        self.feature_layer = args.feature_layer
        self.embedding_layer = args.embedding_layer

        # Default hyperparameters
        self.z_value = -0.52
        self.Ni = -0.2
        self.Nj = 0.2
        self.alpha_t = 0.65
        self.cosine_thr = 0.88
        
        # Build SAM and pseudo backbone
        self.sam_backbone, self.pseudo_backbone = load_backbones(args.sam_backbone, args.pseudo_backbone)
    
        # Build automatic mask generator
        self.automatic_mask_generator = SamAutomaticMaskGenerator(
            model=self.sam_backbone,
            points_per_side=args.points_per_side,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=0
        )

        # Build pseudo generator
        self.pseudo_generator = PseudoGenerator(
            feature_layer=self.feature_layer,
            embedding_layer=self.embedding_layer,
            img_size=self.img_size,
            backbone=self.pseudo_backbone
        )
        
    
    def load_img(self, img_path):
        """Load and preprocess an image (RGB and grayscale)."""
        
        # Load and resize RGB image
        bgr_img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.resize(rgb_img, self.img_size)
        rgb_img = np.array(rgb_img)

        # Load and resize grayscale image
        gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        gray_img = cv2.resize(gray_img, self.img_size) / 255.
        gray_img = np.array(gray_img)

        # Transform RGB image to tensor
        input_tensor = self.transform()(rgb_img).unsqueeze(0)

        return rgb_img, gray_img, input_tensor


    def transform(self):
        """Return composed torchvision transform for RGB image preprocessing."""
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(self.img_size),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])
        
        
    def get_skewness_and_type(self, key, query, value, img_t1, flag):
        """Calculate skewness, type, and moderate region mask based on feature similarity map."""

        # Select similarity feature map
        if self.feature_facet == 'key':
            sim_map = key
        elif self.feature_facet == 'query':
            sim_map = query
        else:
            sim_map = value

        sim_map = sim_map.detach().cpu().numpy()[0]

        # Flatten relevant regions based on flag
        if flag:
            oov_idx = np.where(np.any(img_t1 != [0, 0, 0], axis=-1))
            flat_sim_map = sim_map[oov_idx].flatten()
        else:
            oov_idx = None
            flat_sim_map = sim_map.flatten()

        # Compute skewness
        skewness = skew(flat_sim_map)

        # Determine skewness type
        if skewness >= self.Nj:
            skew_type = 'Right-skewed'
        elif skewness <= self.Ni:
            skew_type = 'Left-skewed'
        else:
            skew_type = 'Moderate'

        # Generate moderate region mask using z-score thresholding
        mean = np.mean(sim_map)
        std = np.std(sim_map)
        z_score = (sim_map - mean) / std
        moderate_mask = (z_score < self.z_value).astype(np.float32)

        return skewness, skew_type, sim_map, flat_sim_map, moderate_mask, oov_idx


    def threshold(self, skewness):
        """Calculate dynamic threshold based on image size and skewness type."""
        h = self.img_size[0]
        w = self.img_size[1]
        
        # Threshold parameters
        b_left = 0.7
        b_right = 0.05
        s_left = 1.0
        s_right = 0.1
        mu = 2.5e5 / (h * w)
        c = 1.0 / mu**3

        # Threshold calculation based on skewness
        if skewness >= self.Nj:
            threshold = b_right + s_right * skewness * c
        elif skewness <= self.Ni:
            threshold = b_left - s_left * skewness * c
        else:
            threshold = 0.0

        return threshold
    
    
    def adaptive_threshold_function(self, flat_sim_map, skewness):
        """Detect outliers in the similarity map using MAD-based adaptive thresholding."""

        # Compute Median Absolute Deviation (MAD)
        median = np.median(flat_sim_map)
        mad = np.median(np.abs(flat_sim_map - median))

        # Compute modified z-scores
        modified_z_scores = 0.6745 * (flat_sim_map - median) / mad

        # Determine threshold from skewness
        threshold = self.threshold(skewness)

        # Identify outliers
        outliers = modified_z_scores < (-1.0 * threshold)

        return outliers

    
    def forward(self, img_t0_path, img_t1_path):
        '''Generate final change mask from a pair of input images.'''

        # Load and preprocess input images
        img_t0, gray_img_t0, input_t0 = self.load_img(img_t0_path)
        img_t1, gray_img_t1, input_t1 = self.load_img(img_t1_path)

        # Generate class-agnostic object proposals
        masks_t0 = self.automatic_mask_generator.generate(img_t0)
        masks_t1 = self.automatic_mask_generator.generate(img_t1)

        # Coarse alignment
        aligned_img_t1, H, flag = coarse_transform(
            self.dataset, self.img_size, img_t0, img_t1, gray_img_t0, gray_img_t1
        )
        if self.dataset == 'Remote_Sensing':
            flag = False

        # Optional realignment and regeneration for aligned image
        if flag:
            img_t1 = np.array(aligned_img_t1)
            input_t1 = self.transform()(img_t1).unsqueeze(0)
            if self.dataset == 'ChangeSim':
                masks_t1 = self.automatic_mask_generator.generate(img_t1)

        ####################################################
        # Initial Pseudo Mask Generation
        ####################################################
        
        # Feature embedding and similarity
        inputs = torch.cat([input_t0, input_t1], dim=1).to(device='cuda')
        embed_t0, embed_t1, key, query, value = self.pseudo_generator(inputs)

        # Analyze similarity distribution
        skewness, type, sim_map, flat_sim_map, moderate_mask, oov_idx = self.get_skewness_and_type(
            key, query, value, img_t1, flag
        )

        # Thresholding and outlier detection
        outliers = self.adaptive_threshold_function(flat_sim_map, skewness)
        binary_mask_outliers = np.zeros_like(sim_map, dtype=np.uint8)

        if flag:
            binary_mask_outliers[oov_idx[0][outliers], oov_idx[1][outliers]] = 1
        else:
            outliers_reshaped = outliers.reshape(self.img_size)
            binary_mask_outliers[outliers_reshaped] = 1
        
        # Refine noise and out-of-view (OOV) regions
        if self.dataset == 'VL_CMU_CD':
            oov_mask = np.all(img_t0 == [0, 0, 0], axis=-1)
            binary_mask_outliers[oov_mask] = 0
            moderate_mask[oov_mask] = 0

        if flag:
            warped_oov_mask = np.all(img_t1 == [0, 0, 0], axis=-1)
            binary_mask_outliers[warped_oov_mask] = 0
            moderate_mask[warped_oov_mask] = 0

            # Dilate warped OOV mask to suppress border noise
            padding_size = 10
            kernel_size = 2 * padding_size + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            warped_oov_mask_uint8 = warped_oov_mask.astype(np.uint8) * 255
            dilated_mask = cv2.dilate(warped_oov_mask_uint8, kernel, iterations=1).astype(bool)

            binary_mask_outliers[dilated_mask] = 0
            moderate_mask[dilated_mask] = 0

            # Inverse warp
            if self.dataset not in ['ChangeSim', 'VL_CMU_CD']:
                H_inv = np.linalg.inv(H)
                binary_mask_outliers = cv2.warpPerspective(binary_mask_outliers, H_inv, self.img_size)
                moderate_mask = cv2.warpPerspective(moderate_mask, H_inv, self.img_size)

        # Select final pseudo mask based on skewness type
        if type in ['Left-skewed', 'Right-skewed']:
            initial_pseudo_mask = binary_mask_outliers
        else:
            initial_pseudo_mask = moderate_mask
        
        # Refine initial pseudo mask (small region removal and morphological smoothing)
        initial_pseudo_mask = initial_pseudo_mask.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        initial_pseudo_mask = cv2.morphologyEx(initial_pseudo_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(initial_pseudo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                cv2.drawContours(initial_pseudo_mask, [cnt], -1, (0, 0, 0), thickness=cv2.FILLED)

        ####################################################
        # Geometric-Semantic Mask Matching 
        ####################################################

        mask_idx_t0 = []
        mask_idx_t1 = []

        # Geometric + Semantic matching: masks_t0
        for i in range(len(masks_t0)):
            iou, overlap_mask = calculate_iou(initial_pseudo_mask, masks_t0[i]['segmentation'])
            if iou >= self.alpha_t:
                mask_embedding_t0 = embed_t0[overlap_mask].mean(axis=0)
                mask_embedding_t1 = embed_t1[overlap_mask].mean(axis=0)
                cosine_similarity = torch.nn.functional.cosine_similarity(mask_embedding_t0, mask_embedding_t1, dim=0)
                if cosine_similarity < self.cosine_thr:
                    mask_idx_t0.append(i)

        x = np.zeros_like(initial_pseudo_mask)
        for j in mask_idx_t0:
            x = np.logical_or(x, masks_t0[j]['segmentation'])

        # Geometric + Semantic matching: masks_t1
        if not self.dataset_bias:
            for k in range(len(masks_t1)):
                iou, overlap_mask = calculate_iou(initial_pseudo_mask, masks_t1[k]['segmentation'])
                if iou >= self.alpha_t:
                    mask_embedding_t0 = embed_t0[overlap_mask].mean(axis=0)
                    mask_embedding_t1 = embed_t1[overlap_mask].mean(axis=0)
                    cosine_similarity = torch.nn.functional.cosine_similarity(mask_embedding_t0, mask_embedding_t1, dim=0)
                    if cosine_similarity < self.cosine_thr:
                        mask_idx_t1.append(k)

            y = np.zeros_like(initial_pseudo_mask)
            for l in mask_idx_t1:
                y = np.logical_or(y, masks_t1[l]['segmentation'])

            final_change_mask = np.logical_or(x, y)
        else:
            final_change_mask = x
            
        # Resize final_change_mask to output-size and ensure binary output
        final_change_mask = final_change_mask.astype(np.uint8) * 255  # ensure dtype and scale for cv2
        final_change_mask = cv2.resize(final_change_mask, self.output_size, interpolation=cv2.INTER_LINEAR)
        final_change_mask = (final_change_mask > 127).astype(np.uint8)  # threshold to binary

        return final_change_mask
                
        
        
        