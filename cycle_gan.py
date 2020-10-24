import torch 
from torch import nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        x = self.bn(x)
        x = F.relu(x)
        return x 

class CycleGenerator(nn.Module):
    def __init__(self, conv_dim=64, init_zero_weights=False):
        super(CycleGenerator, self).__init__()
        # encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, conv_dim//2, kernel_size=(4,4), stride=2, padding=(2,2)),
            nn.BatchNorm2d(conv_dim//2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv_dim//2, conv_dim, kernel_size=(4,4), stride=2, padding=(2,2)),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU()
        )
        
        # resnet transformation
        self.resnet_block = ResNetBlock(in_channels=conv_dim, out_channels=conv_dim)

        # decoder
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(conv_dim, conv_dim//2, kernel_size=(4,4), stride=2, padding=(2,2)),
            nn.BatchNorm2d(conv_dim//2),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(conv_dim//2, 3, kernel_size=(4,4), stride=2, padding=(2,2)),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.resnet_block(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        return x 

class CycleDiscriminator(nn.Module):
    def __init__(self,):
        super(CycleDiscriminator, self).__init__()
        

        
