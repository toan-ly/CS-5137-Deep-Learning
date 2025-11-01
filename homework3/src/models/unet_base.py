import torch 
import torch.nn as nn
import torch.nn.functional as F

def get_activation(name: str = 'relu') -> nn.Module:
    return {
        'relu': nn.ReLU(inplace=True),
        'leaky_relu': nn.LeakyReLU(0.1, inplace=True),
        'elu': nn.ELU(inplace=True),
        'gelu': nn.GELU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'silu': nn.SiLU(),
    }.get(name, nn.ReLU())


class DoubleConv(nn.Module):
    """
    Applies 2 consecutive convolutional layers each followed by
    batch normalization and an activation function.

    Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU

    Use padding=1 to maintain the image size.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        activation: str = 'relu',
        dropout: float = 0.0
    ):
        super(DoubleConv, self).__init__()
        act = get_activation(activation)
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            act,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            act,
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

class ResidualConv(nn.Module):
    """
    Residual convolutional block with two convolutional layers and a residual connection.

    Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> Add Residual -> ReLU
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        activation: str = 'relu',
        dropout: float = 0.0
    ):
        super(ResidualConv, self).__init__()
        act = get_activation(activation)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = act
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else None

        # If in_channels != out_channels, use a 1x1 conv to match dimensions
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(residual)

        out += residual # residual connection
        out = self.activation(out)

        if self.dropout:
            out = self.dropout(out)

        return out
    
class DownBlock(nn.Module):
    """
    Downsampling block that applies DoubleConv followed by MaxPooling.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        activation: str = 'relu',
        dropout: float = 0.0,
        double_conv=DoubleConv
    ):
        super(DownBlock, self).__init__()
        self.double_conv = double_conv(in_channels, out_channels, activation=activation, dropout=dropout)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        feat = self.double_conv(x) # for skip connection when upsampling
        down = self.maxpool(feat) # for next lower level
        return feat, down

class UpBlock(nn.Module):
    """
    Upsampling block that applies Transposed Convolution followed by DoubleConv.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        activation: str = 'relu',
        dropout: float = 0.0,
        up_mode: str = 'transpose',  # 'transpose' or 'bilinear'
        double_conv=DoubleConv
    ):
        super(UpBlock, self).__init__()
        self.up_mode = up_mode
        # Note: the output channels after upsampling should be in_channels // 2
        if up_mode == 'transpose':
            self.up_conv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        elif up_mode == 'bilinear':
            self.up_conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                # Use 1x1 conv to reduce the number of channels since bilinear doesn't change channels
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
            )
        else:
            raise ValueError(f"Unsupported up_mode: {up_mode}")

        self.double_conv = double_conv(in_channels, out_channels, activation=activation, dropout=dropout)

    def forward(self, x, skip_connection):
        x = self.up_conv(x)
        x = self._crop_and_concat(x, skip_connection)
        x = self.double_conv(x)
        return x

    def _crop_and_concat(self, x, skip):
        """
        Crop skip to match x's size and concatenate along channel dimension.
        """
        diff_y = skip.size(2) - x.size(2) # height
        diff_x = skip.size(3) - x.size(3) # width

        # I actually use padding here instead of cropping to avoid losing information
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                      diff_y // 2, diff_y - diff_y // 2])
        return torch.cat([skip, x], dim=1)

class FinalOutput(nn.Module):
    """
    Final output layer to map to desired number of classes.
    It uses a 1x1 convolution.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(FinalOutput, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)