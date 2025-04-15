import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                   
            nn.Conv2d(channels, channels // reduction, 1),  
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),  
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale



class PoseCNNv2(nn.Module):
    def __init__(self, num_classes=20):
        super(PoseCNNv2, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                SEBlock(out_channels),          
                nn.MaxPool2d(2)
            )

        
        self.backbone = nn.Sequential(
            conv_block(3, 48),      # 144×256
            conv_block(48, 96),     # 72×128
            conv_block(96, 192),    # 36×64
            conv_block(192, 384),   # 18×32
            conv_block(384, 768),   # 9×16
        )

        
        self.pool = nn.AdaptiveAvgPool2d(1)  # → (B, 512, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(768, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )


    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x
