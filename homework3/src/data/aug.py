from monai.transforms import (
    LoadImageD, EnsureChannelFirstD, Compose, ResizeD,
    ScaleIntensityRangeD, RandFlipD, RandRotateD,
    RandZoomD, EnsureTypedD, AsDiscreteD, ToTensorD,
    RandAdjustContrastD
)

def get_transforms(
    im_size=512,
    normalize=True,
    is_train=True
):
    keys = ['image', 'mask']
    transforms = [
        LoadImageD(keys=keys),
        EnsureChannelFirstD(keys=keys),
        ResizeD(keys=keys, spatial_size=(im_size, im_size), mode=("bilinear", "nearest"))
    ]
    
    if normalize:
        transforms += [
            ScaleIntensityRangeD(
                keys=['image'],
                a_min=0,
                a_max=255,
                b_min=0.0,
                b_max=1.0,
                clip=True
            )
        ]
    
    if is_train:
        transforms += [
            RandFlipD(keys=keys, prob=0.5, spatial_axis=0), # horizontal flip
            RandFlipD(keys=keys, prob=0.5, spatial_axis=1), # vertical flip
            RandRotateD(keys=keys, range_x=0.087, prob=0.3), # 5 degrees
            RandZoomD(keys=keys, min_zoom=0.9, max_zoom=1.1, prob=0.3),
            RandAdjustContrastD(keys=['image'], prob=0.3, gamma=(0.9, 1.1)),
        ]
    
    transforms += [
        EnsureTypedD(keys=keys),
        AsDiscreteD(keys=['mask'], threshold=0.5),
        # ToTensorD(keys=keys),
    ]
    
    return Compose(transforms)