import torch.nn as nn

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
                layers.append(nn.Dropout2d(0.3)) 
            return nn.Sequential(*layers)

        self.conv1 = conv_block(3, 32)
        self.conv2 = conv_block(32, 64)
        self.conv3 = conv_block(64, 128, use_dropout=False)
        self.conv4 = conv_block(128, 256, use_dropout=True)
        self.conv5 = conv_block(256, 512, use_dropout=True)

        # Enhanced skip projection with double downsampling
        self.skip_proj = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1),  # Channel adjustment
            nn.MaxPool2d(2),  # First downsampling (72x128 -> 36x64)
            nn.MaxPool2d(2)   # Second downsampling (36x64 -> 18x32)
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
        x1 = self.conv1(x)       # (32, 144, 256)
        x2 = self.conv2(x1)       # (64, 72, 128)
        x3 = self.conv3(x2)       # (128, 36, 64)
        x4 = self.conv4(x3)       # (256, 18, 32)

        skip = self.skip_proj(x2) # (256, 18, 32)
        x4 = x4 + skip            # Now compatible shapes
        
        x5 = self.conv5(x4)       # (512, 9, 16)
        x_pooled = self.pool(x5)  # (512, 1, 1)
        return self.classifier(x_pooled)