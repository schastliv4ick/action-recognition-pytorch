import torch.nn as nn


class PoseCNNsc_13_24_35(nn.Module):
    def __init__(self, num_classes=20):
        super(PoseCNNsc_13_24_35, self).__init__()

        def conv_block(in_channels, out_channels, use_dropout=False):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ]
            if use_dropout:
                layers.append(nn.Dropout2d(0.4))
            return nn.Sequential(*layers)

        self.conv1 = conv_block(3, 32)  # Output: (32, H/2, W/2)
        self.conv2 = conv_block(32, 64)  # Output: (64, H/4, W/4)
        self.conv3 = conv_block(64, 128, use_dropout=False)  # Output: (128, H/8, W/8)
        self.conv4 = conv_block(128, 256, use_dropout=True)  # Output: (256, H/16, W/16)
        self.conv5 = conv_block(256, 512, use_dropout=True)  # Output: (512, H/32, W/32)

        # Skip connection from conv1 to conv3
        self.skip1_proj = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=1),
            nn.MaxPool2d(kernel_size=4, stride=4)  # Downsample H/2 -> H/8, W/2 -> W/8
        )

        # Skip connection from conv2 to conv4
        self.skip2_proj = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample H/4 -> H/8, W/4 -> W/8
        )

        # Skip connection from conv3 to conv5
        self.skip3_proj = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=1),
            nn.MaxPool2d(kernel_size=4, stride=4)  # Downsample H/8 -> H/32, W/8 -> W/32
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x1 = self.conv1(x)  # (32, H/2, W/2)
        x2 = self.conv2(x1)  # (64, H/4, W/4)
        x3 = self.conv3(x2)  # (128, H/8, W/8)

        # Skip connection 1 -> 3
        skip1 = self.skip1_proj(x1)  # (128, H/8, W/8)
        x3 = x3 + skip1

        # Skip connection 2 -> 4
        skip2 = self.skip2_proj(x2)  # (256, H/8, W/8)
        x4 = self.conv4(x3)  # (256, H/16, W/16)
        # Необходимо выполнить еще одно понижение разрешения для skip2, чтобы соответствовать x4
        skip2_downsampled = nn.MaxPool2d(kernel_size=2, stride=2)(skip2)  # (256, H/16, W/16)
        x4 = x4 + skip2_downsampled

        x5 = self.conv5(x4)  # (512, H/32, W/32)

        # Skip connection 3 -> 5
        skip3 = self.skip3_proj(x3)  # (512, H/32, W/32)
        x5 = x5 + skip3

        x_pooled = self.pool(x5)  # (512, 1, 1)
        return self.classifier(x_pooled)
