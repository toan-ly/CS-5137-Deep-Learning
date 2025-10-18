import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        output_dim: int = 10,
        batch_norm: bool = True,
        fc_layers: list = [128],
        dropout: float = 0.0,
    ):
        super(VGG, self).__init__()

        self.vgg16_cfg = [
            64, 64, 'M', 
            128, 128, 'M', 
            256, 256, 256, 'M', 
            512, 512, 512, 'M', 
            512, 512, 512, # 'M' # Removed this max pooling for mnist 28x28
        ]

        self.feature_extractor = self._make_layers(input_channels, self.vgg16_cfg, batch_norm)
        self.flatten = nn.Flatten()
        fc_layers = [512] + fc_layers + [output_dim]
        fc_blocks = []
        for i in range(len(fc_layers) - 2):
            fc_blocks.append(nn.Linear(fc_layers[i], fc_layers[i+1]))
            fc_blocks.append(nn.ReLU())
            if dropout > 0:
                fc_blocks.append(nn.Dropout(dropout))
        # Final output layer
        fc_blocks.append(nn.Linear(fc_layers[-2], fc_layers[-1]))
        self.fc = nn.Sequential(*fc_blocks)

    def forward(self, x):
        x = self.feature_extractor(x) # Include Global Average Pooling
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def _make_layers(self, input_channels, cfg, batch_norm=True):
        blocks = []
        in_channels = input_channels
        for h in cfg:
            if h == 'M': # Max Pooling
                blocks += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else: # Convolutional layer
                conv = nn.Conv2d(in_channels, h, kernel_size=3, padding=1)
                blocks += [conv]
                if batch_norm:
                    blocks += [nn.BatchNorm2d(h)]
                blocks += [nn.ReLU(inplace=True)]
                in_channels = h
        blocks += [nn.AdaptiveAvgPool2d((1, 1))]
        return nn.Sequential(*blocks)