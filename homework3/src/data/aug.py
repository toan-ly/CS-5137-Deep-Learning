from monai.transforms import (
    LoadImageD, EnsureChannelFirstD, Compose, ResizeD,
    ScaleIntensityRangeD, RandFlipD, RandRotateD, RandRotate90D,
    RandZoomD, EnsureTypeD, AsDiscreteD, RandGaussianNoiseD, 
    RandCropByPosNegLabelD, LambdaD, RandAdjustContrastD, RandHistogramShiftD,
    IdentityD
)

from .clahe import apply_clahe

def get_transforms(
    im_size=512,
    use_green_channel=False,
    is_train=True,
    use_patch=False,
    patch_size=256
):
    if is_train:
        return train_transforms(im_size, use_green_channel, use_patch, patch_size)
    else:
        return val_transforms(im_size, use_green_channel)


def train_transforms(im_size=512, use_green_channel=False, use_patch=False, patch_size=256):
    keys = ('image', 'mask')
    mode = ('bilinear', 'nearest')

    if use_patch:
        num_samples = (im_size // patch_size) * 4

    return Compose([
        LoadImageD(keys=keys),
        EnsureChannelFirstD(keys=keys),
        ResizeD(keys=keys, spatial_size=(im_size, im_size), mode=mode),
        LambdaD(keys=['image'], func=lambda x: x[1:2, ...] if use_green_channel else x),
        LambdaD(keys=['image'], func=lambda x: apply_clahe(x, clip_limit=2.0, tile_grid_size=(8,8), prob=0.8)),
        ScaleIntensityRangeD(keys=['image'], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        RandCropByPosNegLabelD(
            keys=keys,
            label_key='mask',
            spatial_size=(patch_size, patch_size),
            pos=3, neg=1,
            num_samples=num_samples,
            image_key='image',
            image_threshold=0,
        ) if use_patch else IdentityD(keys=keys),
        # Flip, Rotate, Zoom
        RandFlipD(keys=keys, prob=0.5, spatial_axis=0), # horizontal flip
        RandFlipD(keys=keys, prob=0.5, spatial_axis=1), # vertical flip
        RandRotate90D(keys=keys, prob=0.5, max_k=3),
        RandRotateD(keys=keys, range_x=0.17, prob=0.25, mode=mode),
        RandZoomD(keys=keys, min_zoom=0.97, max_zoom=1.03, prob=0.25, mode=mode),

        # Photometric
        RandAdjustContrastD(keys=["image"], prob=0.25, gamma=(0.9, 1.1)),
        RandHistogramShiftD(keys=["image"], prob=0.25, num_control_points=6),
        RandGaussianNoiseD(keys=["image"], prob=0.1, mean=0.0, std=0.005),

        EnsureTypeD(keys=keys),
        AsDiscreteD(keys=['mask'], threshold=0.5),
    ])

def val_transforms(im_size=512, use_green_channel=False):
    keys = ('image', 'mask')
    mode = ('bilinear', 'nearest')

    return Compose([
        LoadImageD(keys=keys),
        EnsureChannelFirstD(keys=keys),
        ResizeD(keys=keys, spatial_size=(im_size, im_size), mode=mode),
        LambdaD(keys=['image'], func=lambda x: x[1:2, ...] if use_green_channel else x),
        LambdaD(keys=['image'], func=lambda x: apply_clahe(x, clip_limit=2.0, tile_grid_size=(8,8), prob=0.0)),
        ScaleIntensityRangeD(keys=['image'], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        EnsureTypeD(keys=keys),
        AsDiscreteD(keys=['mask'], threshold=0.5),
    ])

