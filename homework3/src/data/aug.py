from monai.transforms import (
    LoadImageD, EnsureChannelFirstD, Compose, ResizeD,
    ScaleIntensityRangeD, RandFlipD, RandRotateD, RandRotate90D,
    RandZoomD, EnsureTypeD, AsDiscreteD, RandGaussianNoiseD, 
    RandCropByPosNegLabelD, LambdaD, RandAdjustContrastD, RandHistogramShiftD,
    RandShiftIntensityD, RandGaussianSmoothD, RandGridDistortionD, RandScaleIntensityD,
)
import numpy as np

from .clahe import apply_clahe, append_clahe
from .small_vessel import create_new_mask, extract_small_vessel

def get_transforms(
    im_size=512,
    use_green_channel=False,
    is_train=True,
    # use_patch=False,
    patch_size=128
):
    if is_train:
        return train_transforms(im_size, use_green_channel, patch_size)
    else:
        return val_transforms(im_size, use_green_channel)


def train_transforms(im_size=512, use_green_channel=False, patch_size=128):
    keys = ('image', 'mask')
    mode = ('bilinear', 'nearest')

    num_samples = (im_size // patch_size) * 2 # 4 for 256x256 and 8 for 128x128

    return Compose([
        LoadImageD(keys=keys),
        EnsureChannelFirstD(keys=keys),
        LambdaD(keys=['mask'], func=lambda x: (x > 0.5).astype(np.uint8)),
        LambdaD(keys=['mask'], func=lambda x: create_new_mask(x, extract_small_vessel(x, kernel_size=5, struct_elem='cross'))),

        LambdaD(keys=['image'], func=lambda x: x[1:2, ...] if use_green_channel else x),
        # LambdaD(keys=['image'], func=lambda x: apply_clahe(x, clip_limit=2.0, tile_grid_size=(8,8), prob=1.0)),
        LambdaD(keys=['image'], func=lambda x: append_clahe(x, clip_limit=2.0, tile_grid_size=(8,8), prob=1.0, use_green_channel=True)),

        ScaleIntensityRangeD(keys=['image'], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),

        # ------------- Geometric -------------
        # Basic flips and rotations
        RandFlipD(keys=keys, prob=0.5, spatial_axis=0), # horizontal flip
        RandFlipD(keys=keys, prob=0.5, spatial_axis=1), # vertical flip
        RandRotate90D(keys=keys, prob=0.5, max_k=3),
        RandRotateD(keys=keys, range_x=np.pi/6, prob=0.5, mode=mode),
        # RandZoomD(keys=keys, min_zoom=0.8, max_zoom=1.2, prob=0.5, mode=mode),

        # Grid distortion
        RandGridDistortionD(keys=keys, prob=0.3, distort_limit=(-0.05, 0.05), mode=mode),

        # ------------- Photometric -------------
        # Intensity and contrast adjustments
        RandAdjustContrastD(keys=["image"], prob=0.3, gamma=(0.6, 1.4)),
        RandScaleIntensityD(keys=["image"], factors=(-0.3, 0.3), prob=0.3),
        # RandHistogramShiftD(keys=["image"], prob=0.25, num_control_points=(3, 10)),
        # RandShiftIntensityD(keys=["image"], offsets=0.1, prob=0.5),

        # RandGaussianSmoothD(keys=["image"], prob=0.3, sigma_x=(0.6, 1.2)),
        RandGaussianNoiseD(keys=["image"], prob=0.3, mean=0.0, std=0.02),

        # Patch cropping
        RandCropByPosNegLabelD(
            keys=keys,
            label_key='mask',
            spatial_size=(patch_size, patch_size),
            pos=6, neg=1,
            num_samples=num_samples,
            image_key='image',
            image_threshold=0,
        ),

        LambdaD(keys=['mask'], func=lambda x: x.astype(np.uint8)),
        EnsureTypeD(keys=keys),
    ])

def val_transforms(im_size=512, use_green_channel=False):
    keys = ('image', 'mask')
    mode = ('bilinear', 'nearest')

    return Compose([
        LoadImageD(keys=keys),
        EnsureChannelFirstD(keys=keys),

        LambdaD(keys=['mask'], func=lambda x: (x > 0.5).astype(np.uint8)),
        LambdaD(keys=['mask'], func=lambda x: create_new_mask(x, extract_small_vessel(x, kernel_size=5, struct_elem='cross'))),
        LambdaD(keys=['image'], func=lambda x: x[1:2, ...] if use_green_channel else x),
        # LambdaD(keys=['image'], func=lambda x: apply_clahe(x, clip_limit=2.0, tile_grid_size=(8,8), prob=1.0)),
        LambdaD(keys=['image'], func=lambda x: append_clahe(x, clip_limit=2.0, tile_grid_size=(8,8), prob=1.0, use_green_channel=True)),
        ScaleIntensityRangeD(keys=['image'], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        EnsureTypeD(keys=keys),
    ])

