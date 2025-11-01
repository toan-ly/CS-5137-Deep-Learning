import os
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from monai.transforms import (
    Compose, LoadImageD, EnsureChannelFirstD, ResizeD, EnsureTypeD,
    RandFlipD, RandRotateD, RandRotate90D, RandZoomD,
    RandAdjustContrastD, RandHistogramShiftD,
    RandGaussianNoiseD, RandGaussianSmoothD, RandCropByPosNegLabelD,
    LambdaD
)

from monai.data import Dataset, DataLoader
from monai.utils import set_determinism

from src.data.clahe import apply_clahe 

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data"
IMAGE_PATH = DATA_ROOT / "train/images/20.png"
MASK_PATH = DATA_ROOT / "train/masks/20.png"
IM_SIZE = 512
USE_GREEN = True            
N_SAMPLES_PER_AUG = 4       
SAVE_DIR = None             

# set_determinism(seed=42)

def keep_green_or_first(x):
    return x[1:2, ...] if x.shape[0] >= 2 else x[0:1, ...]

def to_hwc(img_chw):
    if img_chw.ndim == 3:
        c, h, w = img_chw.shape
        if c == 1:
            return img_chw[0]
        else:
            return np.transpose(img_chw, (1, 2, 0))
    return img_chw

def show_grid(pairs, title, save_path=None):
    cols = len(pairs)
    fig, axes = plt.subplots(2, cols, figsize=(4*cols, 8))
    fig.suptitle(title, fontsize=14)

    for i, (img_c, msk_c) in enumerate(pairs):
        img = to_hwc(img_c) 
        msk = to_hwc(msk_c) 

        # image
        ax = axes[0, i] if cols > 1 else axes[0]
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(np.clip(img, 0, 1))
        ax.set_title(f"image #{i+1}")
        ax.axis("off")

        # mask
        ax = axes[1, i] if cols > 1 else axes[1]
        ax.imshow(msk, cmap="gray")
        ax.set_title("mask")
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()

def base_prefix(im_size=IM_SIZE):
    t = [
        LoadImageD(keys=["image", "mask"]),
        EnsureChannelFirstD(keys=["image", "mask"]),
        ResizeD(keys=["image", "mask"], spatial_size=(im_size, im_size), mode=("bilinear", "nearest")),
    ]
    if USE_GREEN:
        t.append(LambdaD(keys="image", func=keep_green_or_first))
    t.append(LambdaD(keys="image", func=lambda x: apply_clahe(x, clip_limit=2.0, tile_grid_size=(8,8), prob=1.0)))
    t.append(EnsureTypeD(keys=["image", "mask"]))
    return Compose(t)

AUG_CANDIDATES = {
    "Flip": Compose([RandFlipD(keys=["image","mask"], prob=1.0, spatial_axis=0)]),
    "FlipV": Compose([RandFlipD(keys=["image","mask"], prob=1.0, spatial_axis=1)]),
    "Rotate90": Compose([RandRotate90D(keys=["image","mask"], prob=1.0, max_k=1)]),
    "RotateSmall": Compose([RandRotateD(keys=["image","mask"], range_x=0.17, prob=1.0, mode=("bilinear","nearest"))]),
    "Zoom": Compose([RandZoomD(keys=["image","mask"], min_zoom=0.95, max_zoom=1.05, prob=1.0, mode=("bilinear","nearest"))]),

    "AdjustContrast": Compose([RandAdjustContrastD(keys=["image"], prob=1.0, gamma=(0.9, 1.1))]),
    "HistShift": Compose([RandHistogramShiftD(keys=["image"], num_control_points=6, prob=1.0)]),
    "GaussianNoise": Compose([RandGaussianNoiseD(keys=["image"], prob=1.0, mean=0.0, std=0.01)]),
    "GaussianSmooth": Compose([RandGaussianSmoothD(keys=["image"], sigma_x=(0.5,1.0), prob=1.0)]),

    "RandCropPosNeg(256)": Compose([
        RandCropByPosNegLabelD(
            keys=["image","mask"], label_key="mask",
            spatial_size=(256,256), pos=3, neg=1, num_samples=1,
            image_key="image", image_threshold=0.0
        )
    ]),
}

def main():
    data = [{"image": IMAGE_PATH, "mask": MASK_PATH}]
    base = base_prefix(IM_SIZE)

    for aug_name, aug in AUG_CANDIDATES.items():
        samples = []
        for i in range(N_SAMPLES_PER_AUG):
            item = data[0].copy()
            d = base(item) 
            out = aug(d)
            if isinstance(out, list):
                for o in out:
                    samples.append((o["image"].numpy(), o["mask"].numpy()))
            else:
                samples.append((out["image"].numpy(), out["mask"].numpy()))

        if len(samples) > N_SAMPLES_PER_AUG:
            random.shuffle(samples)
            samples = samples[:N_SAMPLES_PER_AUG]

        save_path = None
        if SAVE_DIR:
            save_path = os.path.join(SAVE_DIR, f"preview_{aug_name}.png")

        show_grid(samples, title=f"[{aug_name}] preview", save_path=save_path)

if __name__ == "__main__":
    main()