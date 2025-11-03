import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_base import *

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="runpy")

block_map = {
    'base': DoubleConv,
    'residual': ResidualConv,
}

norm_map = {
    'batch': nn.BatchNorm2d,
    # 'layer': nn.LayerNorm,
    'instance': nn.InstanceNorm2d,
}

class UNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 1,
        features: list = [64, 128, 256, 512, 1024], # 4 levels
        activation: str = 'relu',
        dropout: float = 0.0,
        up_mode: str = 'transpose',
        block_type: str = 'base',  # 'base' or 'residual'
        norm_type: str = None,
    ):
        """
        Args:
            n_channels: number of input channels (e.g., 1 for grayscale images)
            n_classes: number of output channels (e.g., number of segmentation classes)
            features: list defining the number of features at each level of the U-Net
            activation: activation function to use in convolutional blocks
            dropout: dropout rate (0.0 means no dropout)
            up_mode: upsampling mode, either 'transpose' for ConvTranspose2d or 'bilinear' for Upsample
        """
        super(UNet, self).__init__()
        assert up_mode in ['transpose', 'bilinear'], "up_mode must be 'transpose' or 'bilinear'"
        assert len(features) >= 2, "features must contain at least 1 encoder level and a bottleneck"

        self.activation = get_activation(activation)

        if block_type not in block_map:
            raise ValueError(f"Unsupported block_type: {block_type}. Supported types are: {list(block_map.keys())}")
        if norm_type is not None and norm_type not in norm_map:
            raise ValueError(f"Unsupported norm_type: {norm_type}. Supported types are: {list(norm_map.keys())}")

        self.norm = norm_map.get(norm_type, None)

        self.double_conv = block_map[block_type]

        # Downsampling path / Encoder
        self.encoder = nn.ModuleList()
        in_c = n_channels
        for feature in features[:-1]:
            self.encoder.append(
                DownBlock(
                    in_channels=in_c,
                    out_channels=feature,
                    activation=activation,
                    dropout=dropout,
                    double_conv=self.double_conv,
                    norm=self.norm
                )
            )
            in_c = feature

        # Bottleneck
        self.bottleneck = self.double_conv(
            in_channels=features[-2],
            out_channels=features[-1],
            activation=activation,
            dropout=dropout,
            norm=self.norm
        )

        # Upsampling path / Decoder
        self.decoder = nn.ModuleList()
        prev_c = features[-1]
        for feature in reversed(features[:-1]):
            self.decoder.append(
                UpBlock(
                    in_channels=prev_c,
                    out_channels=feature,
                    activation=activation,
                    dropout=dropout,
                    up_mode=up_mode,
                    double_conv=self.double_conv,
                    norm=self.norm
                )
            )
            prev_c = feature
        self.final_conv = FinalOutput(features[0], n_classes)

    def forward(self, x):
        skip_connections = []

        for down in self.encoder:
            skip, x = down(x)
            skip_connections.append(skip)

        x = self.bottleneck(x)
        
        for up, skip in zip(self.decoder, reversed(skip_connections)):
            x = up(x, skip)

        return self.final_conv(x)


if __name__ == "__main__":
    x = torch.randn((2, 1, 512, 512))
    preds = UNet(
        n_channels=1, 
        n_classes=2, 
        features=[64, 128, 256, 512], 
        activation='relu', 
        dropout=0.0, 
        up_mode='bilinear',
        norm_type='batch'
    )(x)
    assert preds.shape == (2, 2, 512, 512)
    print('Passed test 1')

    preds = UNet(
        n_channels=1, 
        n_classes=2, 
        features=[64, 128, 256, 512, 1024], 
        activation='leaky_relu', 
        dropout=0.3, 
        up_mode='transpose',
        norm_type='instance'
    )(x)
    assert preds.shape == (2, 2, 512, 512)
    print('Passed test 2')

    preds = UNet(
        n_channels=1,
        n_classes=2,
        features=[64, 128, 256, 512, 1024],
        activation='leaky_relu',
        dropout=0.3,
        up_mode='bilinear',
        block_type='residual',
        norm_type='batch'
    )(x)
    assert preds.shape == (2, 2, 512, 512)
    print('Passed test 3')