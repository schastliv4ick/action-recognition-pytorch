import torch.nn as nn
import torch.nn.functional as F
from torch import cat


class DensePoseCNN(nn.Module):
    def __init__(self, num_classes=20, reduction_channels=16):
        super(DensePoseCNN, self).__init__()

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

        self.conv1 = conv_block(3, 32)
        self.conv2 = conv_block(32, 64)
        self.conv3 = conv_block(64 + 32, 128, use_dropout=False)
        self.conv4 = conv_block(128 + 64 + reduction_channels, 256, use_dropout=True)
        # Change the input channels for conv5 here
        self.conv5 = conv_block(256 + reduction_channels + reduction_channels + reduction_channels, 512,
                                use_dropout=True)

        self.reduce1_3 = nn.Conv2d(32, reduction_channels, kernel_size=1)
        self.reduce2_4 = nn.Conv2d(64, reduction_channels, kernel_size=1)
        self.reduce1_4 = nn.Conv2d(32, reduction_channels, kernel_size=1)
        self.reduce3_5 = nn.Conv2d(128, reduction_channels, kernel_size=1)
        self.reduce2_5 = nn.Conv2d(64, reduction_channels, kernel_size=1)
        self.reduce1_5 = nn.Conv2d(32, reduction_channels, kernel_size=1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)

        x1_resized_3 = F.max_pool2d(x1, kernel_size=2, stride=2)
        x_concat2 = cat([x2, x1_resized_3], dim=1)

        x3 = self.conv3(x_concat2)

        x1_resized_4 = F.max_pool2d(x1, kernel_size=4, stride=4)
        x2_resized_4 = F.max_pool2d(x2, kernel_size=2, stride=2)
        x1_reduced_4 = self.reduce1_4(x1_resized_4)
        x2_reduced_4 = self.reduce2_4(x2_resized_4)
        x_concat3 = cat([x3, x2_resized_4, x1_reduced_4], dim=1)

        x4 = self.conv4(x_concat3)

        x1_resized_5 = F.max_pool2d(x1, kernel_size=8, stride=8)
        x2_resized_5 = F.max_pool2d(x2, kernel_size=4, stride=4)
        x3_resized_5 = F.max_pool2d(x3, kernel_size=2, stride=2)
        x1_reduced_5 = self.reduce1_5(x1_resized_5)
        x2_reduced_5 = self.reduce2_5(x2_resized_5)
        x3_reduced_5 = self.reduce3_5(x3_resized_5)
        x_concat4 = cat([x4, x3_reduced_5, x2_reduced_5, x1_reduced_5], dim=1)

        x5 = self.conv5(x_concat4)

        x_pooled = self.pool(x5)
        return self.classifier(x_pooled)
