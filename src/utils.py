import numpy as np
from skimage import measure

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