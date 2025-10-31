import cv2
import random

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8,8), prob=0.5):
    if random.random() > prob:
        return img

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img = img.copy()
    if img.ndim == 3: # rgb image
        img[:, :, 0] = clahe.apply(img[:, :, 0])
        img[:, :, 1] = clahe.apply(img[:, :, 1])
        img[:, :, 2] = clahe.apply(img[:, :, 2])
    elif img.ndim == 1: # grayscale image or single channel
        img = clahe.apply(img)
    return img

