import numpy as np
import cv2

def extract_small_vessel(gt_mask, kernel_size=3):
    """
    Extract small vessels from the ground truth mask using morphological operations.

    gt_mask: binary mask
    kernel_size: size of the structuring element used for morphological operations
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    gt_mask = gt_mask.astype(np.uint8)
    # Perform morphological "opening" to remove small vessels
    opened = cv2.morphologyEx(gt_mask, cv2.MORPH_OPEN, kernel)

    # Subtract the opened mask from the original mask to get small vessels that were removed
    small_vessel_mask = gt_mask - opened
    small_vessel_mask = (small_vessel_mask > 0).astype(np.uint8)

    return small_vessel_mask

def create_new_mask(gt_mask, small_vessel_mask):
    """
    Create new multi-class mask:
        0: background
        1: original vessels
        2: small vessels

    Args:
        gt_mask: original ground truth binary mask
        small_vessel_mask: binary mask of small vessels
    """
    new_mask = np.zeros_like(gt_mask, dtype=np.uint8)
    new_mask[gt_mask > 0] = 1  # Label original vessels as 1
    new_mask[small_vessel_mask > 0] = 2  # Label small vessels as 2

    return new_mask