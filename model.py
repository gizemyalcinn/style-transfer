import torch
from torch import nn

class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super().__init__()
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride)

    def forward(self, x):
        return self.conv(self.pad(x))


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = ConvLayer(ch, ch, 3, 1)
        self.in1 = nn.InstanceNorm2d(ch, affine=True)
        self.conv2 = ConvLayer(ch, ch, 3, 1)
        self.in2 = nn.InstanceNorm2d(ch, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return out + x


class UpsampleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = ConvLayer(in_ch, out_ch, kernel_size, 1)

    def forward(self, x):
        return self.conv(self.up(x))


class TransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvLayer(3, 32, 9, 1),   nn.InstanceNorm2d(32, affine=True),  nn.ReLU(),
            ConvLayer(32, 64, 3, 2),  nn.InstanceNorm2d(64, affine=True),  nn.ReLU(),
            ConvLayer(64, 128, 3, 2), nn.InstanceNorm2d(128, affine=True), nn.ReLU(),
        )
        self.residuals = nn.Sequential(*[ResidualBlock(128) for _ in range(5)])
        self.decoder = nn.Sequential(
            UpsampleConv(128, 64, 3), nn.InstanceNorm2d(64, affine=True), nn.ReLU(),
            UpsampleConv(64, 32, 3),  nn.InstanceNorm2d(32, affine=True), nn.ReLU(),
            ConvLayer(32, 3, 9, 1),
        )

    def forward(self, x):
        return self.decoder(self.residuals(self.encoder(x)))