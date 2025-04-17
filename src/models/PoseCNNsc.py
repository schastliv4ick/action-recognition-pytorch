import torch.nn as nn
import torch.nn.functional as F

class PoseCNNsc(nn.Module):
    def __init__(self, num_classes=20):
        super(PoseCNNsc, self).__init__()
        
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
        
        self.backbone = nn.Sequential(
        conv_block(3, 32),
        conv_block(32, 64),
        conv_block(64, 128, use_dropout=False),
        conv_block(128, 256, use_dropout=False),
        conv_block(256, 512, use_dropout=False),
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
        x1 = self.backbone[0](x)  
        x2 = self.backbone[1](x1) 
        x3 = self.backbone[2](x2) 
        x4 = self.backbone[3](x3) 
        x5 = self.backbone[4](x4) 

        skip = F.interpolate(x3, size=x5.shape[2:])
        x5 = x5 + skip  

        x = self.pool(x5)
        x = self.classifier(x)
        return x

