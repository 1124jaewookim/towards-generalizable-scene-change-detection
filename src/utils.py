import numpy as np
from skimage import measure
from segment_anything_model import sam_model_registry

import matplotlib.pyplot as plt


# use this function when calculating TC score
def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask2, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou, intersection


def calculate_metric(mask1, mask2):
    mask1 = mask1 > 0. 
    mask2 = mask2 > 0. 
    intersection = np.logical_and(mask1, mask2)
    precision = np.sum(intersection) / (np.sum(mask2) + 1e-9)  
    recall = np.sum(intersection) / (np.sum(mask1) + 1e-9)     
    return precision, recall


def show_mask_new(mask, ax, random_color=False, edge_color='black', contour_thickness=0.0, darker=False):
    # Generate random or fixed color for the mask interior
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    elif darker:
        color = np.array([0, 0, 0, 0.4])  # black with transparency
    else:
        color = np.array([255/255, 80/255, 255/255, 0.6])  # pinkish color with transparency

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    # Show the mask interior
    ax.imshow(mask_image)

    # Find contours for the mask to draw the edges
    contours = measure.find_contours(mask, 0.5)

    for contour in contours:
        # Draw each contour
        ax.plot(contour[:, 1], contour[:, 0], color=edge_color, linewidth=contour_thickness)
        
        
def sanity_check_args(args):
    valid_datasets = ['VL_CMU_CD', 'TSUNAMI', 'ChangeSim', 'ChangeVPR', 'Remote_Sensing', 'Random']
    valid_facets = ['query', 'key', 'value']
    max_layers_by_backbone = {
        'vit_h': 32,
        'vit_l': 24,
        'vit_b': 12
    }
    
    expected_keywords_by_dataset = {
        'VL_CMU_CD': 'VL_CMU_CD',
        'TSUNAMI': 'TSUNAMI',
        'ChangeSim': 'ChangeSim',
        'ChangeVPR': 'ChangeVPR',
        'Remote_Sensing': 'Remote_Sensing',
        'Random': 'Random'
    }

    # Basic validations
    assert args.test_dataset in valid_datasets, \
        f"Invalid test_dataset: {args.test_dataset}. Must be one of {valid_datasets}"
    
    assert args.feature_facet in valid_facets, \
        f"Invalid feature_facet: {args.feature_facet}. Must be one of {valid_facets}"
    
    # Check pseudo_backbone are valid
    assert args.pseudo_backbone in max_layers_by_backbone, \
        f"Invalid pseudo_backbone: {args.pseudo_backbone}. Must be one of {list(max_layers_by_backbone.keys())}"

    # Determine the lowest allowed maximum (conservative bound)
    max_layer = max_layers_by_backbone[args.pseudo_backbone]

    assert 1 <= args.feature_layer <= max_layer, \
        f"Invalid feature_layer: {args.feature_layer}. Must be between 1 and {max_layer} based on selected backbones"

    assert 1 <= args.embedding_layer <= max_layer, \
        f"Invalid embedding_layer: {args.embedding_layer}. Must be between 1 and {max_layer} based on selected backbones"

    # Check dataset path consistency
    if hasattr(args, 'dataset_path') and args.dataset_path:
        dataset_keyword = expected_keywords_by_dataset.get(args.test_dataset)
        assert dataset_keyword.lower() in args.dataset_path.lower(), \
            f"Mismatch between test_dataset='{args.test_dataset}' and dataset_path='{args.dataset_path}'. Expected path to contain keyword: '{dataset_keyword}'\nValid dataset type: {valid_datasets}"
            
            
def load_backbones(backbone_name, pseudo_backbone_name):
    """Load SAM backbones with appropriate checkpoint paths."""
    checkpoint_paths = {
        "vit_h": "./pretrained_weight/sam_vit_h_4b8939.pth",
        "vit_l": "./pretrained_weight/sam_vit_l_0b3195.pth",
        "vit_b": "./pretrained_weight/sam_vit_b_01ec64.pth"
    }

    if backbone_name == pseudo_backbone_name:
        checkpoint = checkpoint_paths[backbone_name]
        backbone = sam_model_registry[backbone_name](checkpoint=checkpoint).to(device='cuda')
        return backbone, backbone
    else:
        sam_backbone = sam_model_registry[backbone_name](
            checkpoint=checkpoint_paths[backbone_name]
        ).to(device='cuda')

        pseudo_backbone = sam_model_registry[pseudo_backbone_name](
            checkpoint=checkpoint_paths[pseudo_backbone_name]
        ).to(device='cuda')

        return sam_backbone, pseudo_backbone
    
    
def visualize_results(rgb_img_t0, rgb_img_t1, final_change_mask, gt=None):
    fig = plt.figure(figsize=(18, 6))

    fig.add_subplot(141)
    plt.title('Image t0')
    plt.imshow(rgb_img_t0)
    plt.axis('off')

    fig.add_subplot(142)
    plt.title('Image t1')
    plt.imshow(rgb_img_t1)
    plt.axis('off')

    fig.add_subplot(143)
    plt.title('Predicted Change Mask')
    plt.imshow(rgb_img_t0)
    show_mask_new(final_change_mask.astype(np.float32), plt.gca())
    plt.axis('off')

    if gt is not None:
        fig.add_subplot(144)
        plt.title('Ground Truth')
        plt.imshow(rgb_img_t0)
        show_mask_new(gt.astype(np.float32), plt.gca())
        plt.axis('off')

    plt.tight_layout()
    plt.show()