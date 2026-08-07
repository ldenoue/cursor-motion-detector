"""MobileNetV4 cursor-hotspot model and CenterNet-style losses."""

import torch
from torch import nn
import torch.nn.functional as F
import timm


class DepthwiseSeparable(nn.Sequential):
    def __init__(self, channels):
        super().__init__(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )


class TemporalCursorNet(nn.Module):
    def __init__(self, pretrained=True, channels=64):
        super().__init__()
        self.backbone = timm.create_model(
            "mobilenetv4_conv_small.e2400_r224_in1k" if pretrained else "mobilenetv4_conv_small",
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3),
        )
        source_channels = self.backbone.feature_info.channels()
        self.lateral = nn.ModuleList([nn.Conv2d(c, channels, 1) for c in source_channels])
        self.fuse8 = DepthwiseSeparable(channels)
        self.fuse4 = DepthwiseSeparable(channels)
        self.shared = DepthwiseSeparable(channels)
        self.heatmap = nn.Conv2d(channels, 1, 1)
        self.offset = nn.Conv2d(channels, 2, 1)
        nn.init.constant_(self.heatmap.bias, -4.6)

    def forward(self, x):
        c4, c8, c16 = self.backbone(x)
        p16 = self.lateral[2](c16)
        p8 = self.fuse8(self.lateral[1](c8) + F.interpolate(p16, size=c8.shape[-2:], mode="bilinear", align_corners=False))
        p4 = self.fuse4(self.lateral[0](c4) + F.interpolate(p8, size=c4.shape[-2:], mode="bilinear", align_corners=False))
        features = self.shared(p4)
        return self.heatmap(features), self.offset(features)


def focal_heatmap_loss(logits, target):
    prediction = logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    positives = target.eq(1).float()
    negatives = target.lt(1).float()
    negative_weight = (1 - target).pow(4)
    positive_loss = -(prediction.log() * (1 - prediction).pow(2) * positives).sum()
    negative_loss = -((1 - prediction).log() * prediction.pow(2) * negative_weight * negatives).sum()
    count = positives.sum().clamp(min=1)
    return (positive_loss + negative_loss) / count


def temporal_cursor_loss(heatmap_logits, offsets, batch):
    heatmap = focal_heatmap_loss(heatmap_logits, batch["heatmap"])
    mask = batch["mask"]
    offset = F.smooth_l1_loss(offsets * mask, batch["offset"] * mask, reduction="sum") / mask.sum().clamp(min=1)
    return heatmap + offset, heatmap.detach(), offset.detach()


class TemporalCursorExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        heatmap, offset = self.model(x)
        return heatmap.sigmoid(), offset
