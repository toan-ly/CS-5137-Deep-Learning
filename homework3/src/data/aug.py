from monai.transforms import (
    LoadImageD, EnsureChannelFirstD, Compose, ResizeD,
    ScaleIntensityRangeD, RandFlipD, RandRotateD, RandRotate90D,
    RandZoomD, EnsureTypeD, AsDiscreteD, RandGaussianNoiseD, 
    RandCropByPosNegLabelD, LambdaD
)

from .clahe import apply_clahe

def get_transforms(
    im_size=512,
    use_green_channel=False,
    is_train=True
):
    keys = ['image', 'mask']
    mode = ('bilinear', 'nearest')
    transforms = [
        LoadImageD(keys=keys),
        EnsureChannelFirstD(keys=keys),
        ResizeD(keys=keys, spatial_size=(im_size, im_size), mode=mode),
        LambdaD(keys=['image'], func=lambda x: x[1:2, ...] if use_green_channel else x),
        LambdaD(keys=['image'], func=lambda x: apply_clahe(x, clip_limit=2.0, tile_grid_size=(8,8), prob=0.5)),
        ScaleIntensityRangeD(keys=['image'], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        
    ]
    
    if is_train:
        transforms += [
            RandCropByPosNegLabelD(
                keys=keys,
                label_key='mask',
                spatial_size=(256, 256),
                pos=3,
                neg=1,
                num_samples=8,
                image_key='image',
                image_threshold=0,
            ),
            # Flip
            RandFlipD(keys=keys, prob=0.5, spatial_axis=0), # horizontal flip
            RandFlipD(keys=keys, prob=0.5, spatial_axis=1), # vertical flip
            RandRotate90D(keys=keys, prob=0.5, max_k=3),
            RandRotateD(keys=keys, range_x=0.17, prob=0.25, mode=mode),

            # Zoom
            RandZoomD(keys=keys, min_zoom=0.95, max_zoom=1.05, prob=0.25, mode=mode),
            RandGaussianNoiseD(keys=["image"], prob=0.1, mean=0.0, std=0.01),
        ]
    
    transforms += [
        EnsureTypeD(keys=keys),
        AsDiscreteD(keys=['mask'], threshold=0.5),
    ]
    
    return Compose(transforms)