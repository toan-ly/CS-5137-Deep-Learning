from monai.transforms import (
    LoadImageD, EnsureChannelFirstD, Compose, ResizeD,
    ScaleIntensityRangeD, RandFlipD, RandRotateD, RandRotate90D,
    RandZoomD, EnsureTypeD, AsDiscreteD, ToTensorD,
    RandAdjustContrastD, RandScaleIntensityD,
    RandShiftIntensityD, RandHistogramShiftD, RandGaussianNoiseD,
    RandGaussianSmoothD, RandBiasFieldD,
)

def get_transforms(
    im_size=512,
    is_train=True
):
    keys = ['image', 'mask']
    transforms = [
        LoadImageD(keys=keys),
        EnsureChannelFirstD(keys=keys),
        ResizeD(keys=keys, spatial_size=(im_size, im_size), mode=("bilinear", "nearest")),
        ScaleIntensityRangeD(keys=['image'], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
    ]
    
    if is_train:
        transforms += [
            # Flip
            RandFlipD(keys=keys, prob=0.5, spatial_axis=0), # horizontal flip
            RandFlipD(keys=keys, prob=0.5, spatial_axis=1), # vertical flip
            RandRotate90D(keys=keys, prob=0.5, max_k=3),
            RandRotateD(keys=keys, range_x=0.17, prob=0.3, mode=('bilinear','nearest')), 

            # Zoom
            RandZoomD(keys=keys, min_zoom=0.9, max_zoom=1.1, prob=0.3, mode=('bilinear','nearest')),

            # RandElasticD(
            #     keys=keys,
            #     prob=0.15,
            #     sigma_range=(2, 5),
            #     magnitude_range=(0.5, 1.5),
            #     mode=('bilinear','nearest'),
            # ),

            RandScaleIntensityD(keys=["image"], factors=0.1, prob=0.3),   # +/-10%
            RandShiftIntensityD(keys=["image"], offsets=0.1, prob=0.3),   # shift by 0.1
            RandAdjustContrastD(keys=["image"], prob=0.3, gamma=(0.9, 1.1)),

            # Histogram shape changes (scanner differences)
            RandHistogramShiftD(keys=["image"], prob=0.25, num_control_points=(5, 10)),

            # Very light noise & smoothing for robustness
            RandGaussianNoiseD(keys=["image"], prob=0.2, mean=0.0, std=0.02),
            RandGaussianSmoothD(keys=["image"], prob=0.15, sigma_x=(0.25, 0.75), sigma_y=(0.25, 0.75)),

            # Low-amplitude bias field to mimic illumination vignetting
            RandBiasFieldD(keys=["image"], prob=0.1, coeff_range=(0.0, 0.05)),
        ]
    
    transforms += [
        EnsureTypeD(keys=keys),
        AsDiscreteD(keys=['mask'], threshold=0.5),
        # ToTensorD(keys=keys),
    ]
    
    return Compose(transforms)