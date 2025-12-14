import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self,num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2, padding=0)
        self.fc = nn.Linear(16,num_classes)

    def __call__(self, x,):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        x = x.mean(dim=(2, 3))
        x = self.fc(x)
        
        return x
    
class ConvBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class DownBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_features, out_features, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_features, out_features, 3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        skip = x
        x = self.pool(x)
        return x, skip

class UpBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_features, out_features,
            kernel_size=2, stride=2
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_features * 2, out_features, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_features, out_features, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3):
        super().__init__()

        # Encoder
        self.down1 = DownBlock(in_channels, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
        )

        # Decoder
        self.up3 = UpBlock(512, 256)
        self.up2 = UpBlock(256, 128)
        self.up1 = UpBlock(128, 64)

        # Output
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.out_fc = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        # Encoder
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)

        # Output conv
        x = self.out_conv(x)

        # Global Average Pooling
        x = x.mean(dim=(2, 3))

        # Dense classifier
        x = self.out_fc(x)
        return x