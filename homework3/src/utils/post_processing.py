import numpy as np
from skimage.morphology import binary_opening, binary_closing, disk
from skimage.measure import label, regionprops

def postprocess_mask(mask, radius=3, min_area=10, ecc_keep=0.8):
    """
    Clean up predicted vessel mask

    Args:
        mask: Predicted mask
        closing_radius: Radius for binary closing to fill small gaps
        min_area: Minimum area to keep a connected component
        ecc_keep: Eccentricity threshold to keep elongated structures
    """
    mask = mask.astype(bool)

    # Apply closing operation to fill small gaps 
    # and opening to remove small noise
    if radius > 0:
        mask = binary_closing(mask, footprint=disk(radius))
        mask = binary_opening(mask, footprint=disk(radius))

    # labeled_mask = label(mask)
    # keep = np.zeros_like(mask, dtype=bool)

    # # Keep connected components based on area and eccentricity
    # for r in regionprops(labeled_mask):
    #     if r.area >= min_area or r.eccentricity >= ecc_keep:
    #         keep[labeled_mask == r.label] = True

    # mask = keep
    return mask.astype(np.uint8)