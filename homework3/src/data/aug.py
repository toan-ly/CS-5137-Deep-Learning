from monai.transforms import (
    LoadImageD, EnsureChannelFirstD, Compose, ResizeD,
    ScaleIntensityRangeD, RandFlipD, RandRotateD, RandRotate90D,
    RandZoomD, EnsureTypeD, AsDiscreteD, RandGaussianNoiseD, 
    RandCropByPosNegLabelD, LambdaD, RandAdjustContrastD, RandHistogramShiftD,
    IdentityD, CopyItemsD, DeleteItemsD
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
        # ResizeD(keys=keys, spatial_size=(im_size, im_size), mode=mode),
        LambdaD(keys=['mask'], func=lambda x: (x > 0.5).astype(np.uint8)),
        LambdaD(keys=['mask'], func=lambda x: create_new_mask(x, extract_small_vessel(x, kernel_size=5))),

        LambdaD(keys=['image'], func=lambda x: x[1:2, ...] if use_green_channel else x),
        # LambdaD(keys=['image'], func=lambda x: apply_clahe(x, clip_limit=2.0, tile_grid_size=(8,8), prob=1.0)),
        LambdaD(keys=['image'], func=lambda x: append_clahe(x, clip_limit=2.0, tile_grid_size=(8,8), prob=1.0)),

        ScaleIntensityRangeD(keys=['image'], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),

        # Flip, Rotate, Zoom
        RandFlipD(keys=keys, prob=0.5, spatial_axis=0), # horizontal flip
        RandFlipD(keys=keys, prob=0.5, spatial_axis=1), # vertical flip
        RandRotate90D(keys=keys, prob=0.5, max_k=3),
        RandRotateD(keys=keys, range_x=np.pi/12, prob=0.5, mode=mode),
        RandZoomD(keys=keys, min_zoom=0.9, max_zoom=1.1, prob=0.25, mode=mode),

        # Photometric
        RandAdjustContrastD(keys=["image"], prob=0.25, gamma=(0.7, 1.3)),
        # RandHistogramShiftD(keys=["image"], prob=0.25, num_control_points=6),
        RandGaussianNoiseD(keys=["image"], prob=0.1, mean=0.0, std=0.02),

        RandCropByPosNegLabelD(
            keys=keys,
            label_key='mask',
            spatial_size=(patch_size, patch_size),
            pos=3, neg=1,
            num_samples=num_samples,
            image_key='image',
            image_threshold=0,
        ),

        LambdaD(keys=['mask'], func=lambda x: x.astype(np.uint8)),
        EnsureTypeD(keys=keys),
        # ToTensorD(keys=keys),
        # AsDiscreteD(keys=['mask'], threshold=0.5),
    ])

def val_transforms(im_size=512, use_green_channel=False):
    keys = ('image', 'mask')
    mode = ('bilinear', 'nearest')

    return Compose([
        LoadImageD(keys=keys),
        EnsureChannelFirstD(keys=keys),
        # ResizeD(keys=keys, spatial_size=(im_size, im_size), mode=mode),

        LambdaD(keys=['mask'], func=lambda x: (x > 0.5).astype(np.uint8)),
        LambdaD(keys=['image'], func=lambda x: x[1:2, ...] if use_green_channel else x),
        # LambdaD(keys=['image'], func=lambda x: apply_clahe(x, clip_limit=2.0, tile_grid_size=(8,8), prob=1.0)),
        LambdaD(keys=['image'], func=lambda x: append_clahe(x, clip_limit=2.0, tile_grid_size=(8,8), prob=1.0)),
        ScaleIntensityRangeD(keys=['image'], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        LambdaD(keys=['mask'], func=lambda x: create_new_mask(x, extract_small_vessel(x, kernel_size=3))),
        EnsureTypeD(keys=keys),
        # ToTensorD(keys=keys),
        # AsDiscreteD(keys=['mask'], threshold=0.5),
    ])

