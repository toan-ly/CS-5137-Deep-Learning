import cv2
import random
import torch
import numpy as np

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8,8), prob=0.5, use_green_channel=False):
    if random.random() > prob:
        return img

    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    else:
        img = np.asarray(img)

    assert img.ndim == 3, f"Image shape {img.shape}"

    img = img.copy()
    img = np.transpose(img, (1, 2, 0))  # CHW to HWC
    img = img.astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if use_green_channel: # use green channel only
        img[:, :, 1] = clahe.apply(img[:, :, 1])
    else: # apply to all channels
        for c in range(img.shape[2]):
            img[:, :, c] = clahe.apply(img[:, :, c])
    
    img = np.transpose(img, (2, 0, 1))  # HWC back to CHW for MONAI format 
    return img

