import torch.nn as nn
import torch.nn.functional as F

class PoseCNNv2(nn.Module):
    def __init__(self, num_classes=20):
        super(PoseCNNv2, self).__init__()
        
        def conv_block(in_channels, out_channels, use_dropout=False):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ]
            if use_dropout:
                layers.append(nn.Dropout2d(0.5)) 
            return nn.Sequential(*layers)
        
        self.backbone = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128, use_dropout=False),
            conv_block(128, 256, use_dropout=True),
            conv_block(256, 512, use_dropout=False),
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)  # → (B, 512, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )


    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x
