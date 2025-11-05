import os, math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from monai.transforms import (
    Compose, LoadImageD, EnsureChannelFirstD, ResizeD, EnsureTypeD,
    RandFlipD, RandRotateD, RandRotate90D, RandZoomD,
    RandAdjustContrastD, RandHistogramShiftD,
    RandGaussianNoiseD, RandGaussianSmoothD, RandCropByPosNegLabelD,
    LambdaD
)
from monai.utils import set_determinism
from src.data.clahe import apply_clahe
from src.data.small_vessel import create_new_mask, extract_small_vessel

ROOT = Path(__file__).parent.parent.parent
IMG = ROOT / "data/train/image/20.png"
MASK = ROOT / "data/train/mask/20.png"
IM_SIZE = 512
SAVE = ROOT / "figures/augmentation_previews"; os.makedirs(SAVE, exist_ok=True)
KEYS, MODE = ("image","mask"), ("bilinear","nearest")
# set_determinism(42)

def preprocess(x):
    if hasattr(x, "as_tensor"): x = x.as_tensor()
    if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
    a = np.asarray(x)
    a = np.transpose(a, (1,2,0)).astype(np.float32)  # CHW -> HWC
    mn, mx = a.min(), a.max()
    if mx > mn: 
        a = (a - mn) / (mx - mn)
    if a.ndim == 3 and a.shape[2] == 1: 
        a = a[..., 0]
    return np.clip(a, 0, 1)

def base(use_green: bool):
    return Compose([
        LoadImageD(keys=KEYS),
        EnsureChannelFirstD(keys=KEYS),
        ResizeD(keys=KEYS, spatial_size=(IM_SIZE, IM_SIZE), mode=MODE),
        LambdaD(keys="mask", func=lambda x: (x > 0.5).astype(np.uint8)),
        LambdaD(keys="image", func=lambda x: x[1:2, ...] if use_green else x),
        # LambdaD(keys="image", func=lambda x: apply_clahe(x, clip_limit=2.0, tile_grid_size=(8,8), prob=1.0)),
        LambdaD(keys="mask", func=lambda x: create_new_mask(x, extract_small_vessel(x, kernel_size=7, struct_elem='rect'))),
        EnsureTypeD(keys=KEYS),
    ])

def build_augs():
    return [
        ("Base",            Compose([])),
        # ("FlipH",           Compose([RandFlipD(keys=KEYS, prob=1.0, spatial_axis=0)])),
        # ("FlipV",           Compose([RandFlipD(keys=KEYS, prob=1.0, spatial_axis=1)])),
        # ("Rotate90",        Compose([RandRotate90D(keys=KEYS, prob=1.0, max_k=1)])),
        ("RotateSmall",     Compose([RandRotateD(keys=KEYS, range_x=np.pi/12, prob=1.0, mode=MODE)])),
        ("Zoom", Compose([RandZoomD(keys=KEYS, min_zoom=0.95, max_zoom=1.05, prob=1.0, mode=MODE)])),
        ("AdjustContrast",  Compose([RandAdjustContrastD(keys=["image"], prob=1.0, gamma=(0.7, 1.3))])),
        ("HistShift",       Compose([RandHistogramShiftD(keys=["image"], num_control_points=6, prob=1.0)])),
        ("GaussNoise",      Compose([RandGaussianNoiseD(keys=["image"], prob=1.0, mean=0.0, std=0.02)])),
        ("GaussSmooth",     Compose([RandGaussianSmoothD(keys=["image"], sigma_x=(0.5,1.0), prob=1.0)])),
        ("RandCrop",     Compose([RandCropByPosNegLabelD(
            keys=KEYS, label_key="mask", spatial_size=(128,128),
            pos=3, neg=1, num_samples=13, image_key="image", image_threshold=0.0
        )])),
    ]

def visualize_augmentations(use_green: bool, outfile: str):
    item = {"image": str(IMG), "mask": str(MASK)}
    b = base(use_green)
    augs = build_augs()

    results = []
    for name, aug in augs:
        out = aug(b(item))
        if name == "RandCrop":
            for i, out in enumerate(out):
                results.append((f"{name}_{i+1}", out["image"], out["mask"]))
        else:
            if isinstance(out, list): 
                out = out[0]
            results.append((name, out["image"], out["mask"]))

    cols = 4
    rows = (len(results) + cols - 1) // cols
    fig, axes = plt.subplots(2*rows, cols, figsize=(4*cols, 8*rows))
    axes = np.array(axes).reshape(2*rows, cols)

    for i, (name, img, msk) in enumerate(results):
        r0, c = (i//cols)*2, i%cols
        img_d, msk_d = preprocess(img), preprocess(msk)
        axes[r0, c].imshow(img_d, cmap="gray" if img_d.ndim==2 else None)
        axes[r0, c].set_title(f"{name}\n{img_d.shape[0]}×{img_d.shape[1]}", fontsize=10)
        axes[r0, c].axis("off")

        axes[r0+1, c].imshow(msk_d)
        axes[r0+1, c].set_title("Mask", fontsize=9)
        axes[r0+1, c].axis("off")

    # for i in range(len(results), rows*cols):
    #     r0, c = (i//cols)*2, i%cols
    #     axes[r0, c].axis("off")
    #     axes[r0+1, c].axis("off")

    plt.tight_layout()
    path = os.path.join(SAVE, outfile)
    plt.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")

def main():
    visualize_augmentations(use_green=False, outfile="aug_no_green.png")
    visualize_augmentations(use_green=True,  outfile="aug_with_green.png")

if __name__ == "__main__":
    main()