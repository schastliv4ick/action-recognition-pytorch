from torch import nn


class MiniSimpleton(nn.Module):
    def __init__(self, device):
        super(MiniSimpleton, self).__init__()
        self.relu = nn.ReLU()
        self.maxpool_2 = nn.MaxPool2d(2, 2)
        self.maxpool_4 = nn.MaxPool2d(4, 4)

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.flatten = nn.Flatten()

        # Расчет размеров после сверток и пулинга для входа 3x288x512:
        # conv1(3,8)+pool2: 8x144x256
        # conv2(8,16)+pool2: 16x72x128
        # conv3(16,32)+pool4: 32x18x32
        # Итого: 32 * 18 * 32 = 18432
        self.fc = nn.Linear(32 * 18 * 32, 20)

        self.to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool_2(x)  # 8x144x256

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool_2(x)  # 16x72x128

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool_4(x)  # 32x18x32

        x = self.flatten(x)    # 32*18*32 = 18432
        x = self.fc(x)
        return x