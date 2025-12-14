import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================
# Losses
# ======================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return 1 - dice


# ======================================================
# Building Blocks
# ======================================================
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ---------------- Downsampling ----------------
class DownMP(nn.Module):
    """MaxPool downsampling (exactly invertible)"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        down = self.pool(skip)
        return skip, down


class DownStrided(nn.Module):
    """Strided convolution downsampling (NOT exactly invertible)"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ---------------- Upsampling ----------------
class UpTranspose(nn.Module):
    """Transposed convolution upsampling + safe skip alignment"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)

        # 🔴 CRITICAL FIX: align skip if spatial sizes differ
        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(
                skip,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UpBilinear(nn.Module):
    """Bilinear upsampling + convolution + safe skip alignment"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.pre_conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.conv = ConvBlock(in_c, out_c)

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        x = self.pre_conv(x)

        # 🔴 CRITICAL FIX
        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(
                skip,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return self.bce(logits, targets) + self.dice(logits, targets)


# ======================================================
# MODEL 1: MP + TransposedConv + BCE
# ======================================================
class Model1_MP_TR_BCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = DownMP(3, 64)
        self.d2 = DownMP(64, 128)
        self.d3 = DownMP(128, 256)
        self.d4 = DownMP(256, 512)

        self.bottleneck = ConvBlock(512, 1024)

        self.u1 = UpTranspose(1024, 512)
        self.u2 = UpTranspose(512, 256)
        self.u3 = UpTranspose(256, 128)
        self.u4 = UpTranspose(128, 64)

        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        s1, p1 = self.d1(x)
        s2, p2 = self.d2(p1)
        s3, p3 = self.d3(p2)
        s4, p4 = self.d4(p3)

        b = self.bottleneck(p4)

        x = self.u1(b, s4)
        x = self.u2(x, s3)
        x = self.u3(x, s2)
        x = self.u4(x, s1)

        return self.out(x)  # logits


# ======================================================
# MODEL 2: MP + TransposedConv + Dice
# ======================================================
class Model2_MP_TR_DICE(Model1_MP_TR_BCE):
    pass


# ======================================================
# MODEL 3: StridedConv + TransposedConv + BCE
# ======================================================
class Model3_Strided_TR_BCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = DownStrided(3, 64)
        self.d2 = DownStrided(64, 128)
        self.d3 = DownStrided(128, 256)
        self.d4 = DownStrided(256, 512)

        self.bottleneck = ConvBlock(512, 1024)

        self.u1 = UpTranspose(1024, 512)
        self.u2 = UpTranspose(512, 256)
        self.u3 = UpTranspose(256, 128)
        self.u4 = UpTranspose(128, 64)

        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        s1 = self.d1(x)
        s2 = self.d2(s1)
        s3 = self.d3(s2)
        s4 = self.d4(s3)

        b = self.bottleneck(s4)

        x = self.u1(b, s4)
        x = self.u2(x, s3)
        x = self.u3(x, s2)
        x = self.u4(x, s1)

        return self.out(x)


# ======================================================
# MODEL 4: StridedConv + Bilinear + Dice
# ======================================================
class Model4_Strided_Bilinear_DICE(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = DownStrided(3, 64)
        self.d2 = DownStrided(64, 128)
        self.d3 = DownStrided(128, 256)
        self.d4 = DownStrided(256, 512)

        self.bottleneck = ConvBlock(512, 1024)

        self.u1 = UpBilinear(1024, 512)
        self.u2 = UpBilinear(512, 256)
        self.u3 = UpBilinear(256, 128)
        self.u4 = UpBilinear(128, 64)

        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        s1 = self.d1(x)
        s2 = self.d2(s1)
        s3 = self.d3(s2)
        s4 = self.d4(s3)

        b = self.bottleneck(s4)

        x = self.u1(b, s4)
        x = self.u2(x, s3)
        x = self.u3(x, s2)
        x = self.u4(x, s1)

        return self.out(x)
