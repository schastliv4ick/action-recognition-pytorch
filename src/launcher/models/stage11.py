import torch.nn as nn
import torch.nn.functional as F

class PoseCNNv2_lite_sc_13_24_35(nn.Module):
    def __init__(self, num_classes=20):
        super(PoseCNNv2_lite_sc_13_24_35, self).__init__()

        def conv_block(in_channels, out_channels, use_dropout=False):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ]
            if use_dropout:
                layers.append(nn.Dropout2d(0.2))
            return nn.Sequential(*layers)

        # Создаем отдельные блоки
        self.block1 = conv_block(3, 24)       # 288x512 → 144x256
        self.block2 = conv_block(24, 48)      # 144x256 → 72x128
        self.block3 = conv_block(48, 96)      # 72x128 → 36x64
        self.block4 = conv_block(96, 192, use_dropout=True)  # 36x64 → 18x32
        self.block5 = conv_block(192, 384)    # 18x32 → 9x16
        
        # Слои для skip connections с правильным downsample
        self.skip1_3 = nn.Sequential(
            nn.MaxPool2d(4),  # 144x256 → 36x64
            nn.Conv2d(24, 96, kernel_size=1)
        )
        self.skip2_4 = nn.Sequential(
            nn.MaxPool2d(4),  # 72x128 → 18x32
            nn.Conv2d(48, 192, kernel_size=1)
        )
        self.skip3_5 = nn.Sequential(
            nn.MaxPool2d(4),  # 36x64 → 9x16
            nn.Conv2d(96, 384, kernel_size=1)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(384, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Проход через блоки
        x1 = self.block1(x)  # 24, 144x256
        x2 = self.block2(x1)  # 48, 72x128
        x3 = self.block3(x2)  # 96, 36x64
        
        # Skip connection 1-3
        skip1_3 = self.skip1_3(x1)  # 96, 36x64
        x3 = x3 + skip1_3
        
        x4 = self.block4(x3)  # 192, 18x32
        
        # Skip connection 2-4
        skip2_4 = self.skip2_4(x2)  # 192, 18x32
        x4 = x4 + skip2_4
        
        x5 = self.block5(x4)  # 384, 9x16
        
        # Skip connection 3-5
        skip3_5 = self.skip3_5(x3)  # 384, 9x16
        x5 = x5 + skip3_5
        
        x = self.pool(x5)
        x = self.classifier(x)
        return x