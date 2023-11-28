# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer

from torch.utils.checkpoint import checkpoint
from mmcv.cnn.bricks import ConvModule
from mmdet.models import NECKS


@NECKS.register_module()
class FPN_LSS(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 input_feature_index=(0, 2),
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 lateral=None,
                 use_input_conv=False):
        super(FPN_LSS, self).__init__()
        self.input_feature_index = input_feature_index
        self.extra_upsample = extra_upsample is not None
        self.out_channels = out_channels
        # 用于上采样high-level的feature map
        self.up = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=True)

        channels_factor = 2 if self.extra_upsample else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * channels_factor, kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * channels_factor, out_channels * channels_factor, kernel_size=3,
                      padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor)[1],
            nn.ReLU(inplace=True),
        )

        if self.extra_upsample:
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=extra_upsample, mode='bilinear', align_corners=True),
                nn.Conv2d(out_channels * channels_factor, out_channels, kernel_size=3, padding=1, bias=False),
                build_norm_layer(norm_cfg, out_channels)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
            )

        self.lateral = lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(lateral, lateral, kernel_size=1, padding=0, bias=False),
                build_norm_layer(norm_cfg, lateral)[1],
                nn.ReLU(inplace=True)
            )

    def forward(self, feats):
        """
        Args:
            feats: List[Tensor,] multi-level features
                List[(B, C1, H, W), (B, C2, H/2, W/2), (B, C3, H/4, W/4)]
        Returns:
            x: (B, C_out, 2*H, 2*W)
        """
        x2, x1 = feats[self.input_feature_index[0]], feats[self.input_feature_index[1]]
        if self.lateral:
            x2 = self.lateral_conv(x2)
        x1 = self.up(x1)    # (B, C3, H, W)
        x1 = torch.cat([x2, x1], dim=1)     # (B, C1+C3, H, W)
        x = self.conv(x1)   # (B, C', H, W)
        if self.extra_upsample:
            x = self.up2(x)     # (B, C_out, 2*H, 2*W)
        return x


@NECKS.register_module()
class LSSFPN3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 with_cp=False):
        super().__init__()
        self.up1 = nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=True)
        self.up2 = nn.Upsample(
            scale_factor=4, mode='trilinear', align_corners=True)

        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=dict(type='ReLU', inplace=True))
        self.with_cp = with_cp

    def forward(self, feats):
        """
        Args:
            feats: List[
                (B, C, Dz, Dy, Dx),
                (B, 2C, Dz/2, Dy/2, Dx/2),
                (B, 4C, Dz/4, Dy/4, Dx/4)
            ]
        Returns:
            x: (B, C, Dz, Dy, Dx)
        """
        x_8, x_16, x_32 = feats
        x_16 = self.up1(x_16)       # (B, 2C, Dz, Dy, Dx)
        x_32 = self.up2(x_32)       # (B, 4C, Dz, Dy, Dx)
        x = torch.cat([x_8, x_16, x_32], dim=1)     # (B, 7C, Dz, Dy, Dx)
        if self.with_cp:
            x = checkpoint(self.conv, x)
        else:
            x = self.conv(x)    # (B, C, Dz, Dy, Dx)
        return x




