import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    """
    Encoder Block.
    Consists of one complete layer:
        1. Conv2d
        2. BatchNorm2d
        3. ReLU

    Args:
        Conv2d Args.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x


class ResConvBlock(nn.Module):
    """
    Bottleneck Block
    Consists of one complete layer:
        1. Conv2d
        2. BatchNorm2d
        3. ReLU
        4. Conv2d
        5. BatchNorm2d

    Args:
        Conv2d Args.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        Identity = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)

        return torch.add(Identity, x)


class DecoderBlock(nn.Module):
    """
    Decoder Block.
    Consists of one complete layer:
        1. Upsample/Interpolation
        2. Conv2d
        3. BatchNorm2d
        4. ReLU

    Args:
        Conv2d Args.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="reflect",
    ):
        super().__init__()

        self.stride = stride

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode="reflect",
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Upsample here since not know output size at init stage.
        x = F.interpolate(x, size=x.shape[2] * self.stride)
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x


class runet(nn.Module):
    """
    Full set of RU-net.
    Consists of 4 complete stages:
        1. Encoder
        2. Bottle neck
        3. Decoder
        4. Conv2d

    Args:
        Conv2d Args.

    """

    def __init__(
        self,
        in_channels=3,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        super().__init__()

        # Encoder stages
        # Input shape (N,3,128,128)
        self.enc1 = EncoderBlock(in_channels, 16, kernel_size, stride=2)

        # Input shape (N,16,64,64)
        self.enc2 = EncoderBlock(16, 32, kernel_size, stride=1)

        # Input shape (N,32,64,64)
        self.enc3 = EncoderBlock(32, 64, kernel_size, stride=2)

        # Input shape (N,64,32,32)
        self.enc4 = EncoderBlock(64, 64, kernel_size, stride=1)

        # Input shape (N,64,32,32)
        self.enc5 = EncoderBlock(64, 128, kernel_size, stride=2)

        # Input shape (N,128,16,16)
        self.enc6 = EncoderBlock(128, 128, kernel_size, stride=1)

        # Bottle neck stages
        # Input shape (N,128,16,16)
        self.bn1 = ResConvBlock(128, 128)
        self.bn2 = ResConvBlock(128, 128)
        self.bn3 = ResConvBlock(128, 128)
        self.bn4 = ResConvBlock(128, 128)
        self.bn5 = ResConvBlock(128, 128)
        self.bn6 = ResConvBlock(128, 128)

        # Decoder stages
        # Input shape (N,256,16,16)
        self.dec6 = DecoderBlock(256, 128, kernel_size, stride=1)

        # Input shape (N,256,16,16)
        self.dec5 = DecoderBlock(256, 128, kernel_size, stride=2)

        # Input shape (N,192,32,32)
        self.dec4 = DecoderBlock(192, 64, kernel_size, stride=1)

        # Input shape (N,128,32,32)
        self.dec3 = DecoderBlock(128, 64, kernel_size, stride=2)

        # Input shape (N,96,64,64)
        self.dec2 = DecoderBlock(96, 32, kernel_size, stride=1)

        # Input shape (N,48,64,64)
        self.dec1 = DecoderBlock(48, 16, kernel_size, stride=2)

        # Input shape (N,16,128,128)
        self.output = nn.Conv2d(16, 1, kernel_size, stride=1, padding=1)

    def forward(self, x):
        # Encoder stages
        enc_out_1 = self.enc1(x)
        enc_out_2 = self.enc2(enc_out_1)
        enc_out_3 = self.enc3(enc_out_2)
        enc_out_4 = self.enc4(enc_out_3)
        enc_out_5 = self.enc5(enc_out_4)
        enc_out_6 = self.enc6(enc_out_5)

        # Bottle neck stages
        bn_out_1 = self.bn1(enc_out_6)
        bn_out_2 = self.bn1(bn_out_1)
        bn_out_3 = self.bn1(bn_out_2)
        bn_out_4 = self.bn1(bn_out_3)
        bn_out_5 = self.bn1(bn_out_4)
        bn_out_6 = self.bn1(bn_out_5)

        # Decoder stages
        merge6 = torch.cat((enc_out_6, bn_out_6), 1)
        dec_out_6 = self.dec6(merge6)
        merge5 = torch.cat((enc_out_5, dec_out_6), 1)
        dec_out_5 = self.dec5(merge5)
        merge4 = torch.cat((enc_out_4, dec_out_5), 1)
        dec_out_4 = self.dec4(merge4)
        merge3 = torch.cat((enc_out_3, dec_out_4), 1)
        dec_out_3 = self.dec3(merge3)
        merge2 = torch.cat((enc_out_2, dec_out_3), 1)
        dec_out_2 = self.dec2(merge2)
        merge1 = torch.cat((enc_out_1, dec_out_2), 1)
        dec_out_1 = self.dec1(merge1)

        # Last Conv2d
        output = self.output(dec_out_1)

        return output
