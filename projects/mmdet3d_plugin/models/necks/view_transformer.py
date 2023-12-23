# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import BaseModule, force_fp32
from mmdet3d.models.builder import NECKS
from ...ops import bev_pool_v2
from ..model_utils import DepthNet
from torch.cuda.amp.autocast_mode import autocast
import torch.nn.functional as F


@NECKS.register_module(force=True)
class LSSViewTransformer(BaseModule):
    r"""Lift-Splat-Shoot view transformer with BEVPoolv2 implementation.

    Please refer to the `paper <https://arxiv.org/abs/2008.05711>`_ and
        `paper <https://arxiv.org/abs/2211.17111>`

    Args:
        grid_config (dict): Config of grid alone each axis in format of
            (lower_bound, upper_bound, interval). axis in {x,y,z,depth}.
        input_size (tuple(int)): Size of input images in format of (height,
            width).
        downsample (int): Down sample factor from the input size to the feature
            size.
        in_channels (int): Channels of input feature.
        out_channels (int): Channels of transformed feature.
        accelerate (bool): Whether the view transformation is conducted with
            acceleration. Note: the intrinsic and extrinsic of cameras should
            be constant when 'accelerate' is set true.
        sid (bool): Whether to use Spacing Increasing Discretization (SID)
            depth distribution as `STS: Surround-view Temporal Stereo for
            Multi-view 3D Detection`.
        collapse_z (bool): Whether to collapse in z direction.
    """

    def __init__(
        self,
        grid_config,
        input_size,
        downsample=16,
        in_channels=512,
        out_channels=64,
        accelerate=False,
        sid=False,
        collapse_z=True,
    ):
        super(LSSViewTransformer, self).__init__()
        self.grid_config = grid_config
        self.downsample = downsample
        self.create_grid_infos(**grid_config)
        self.sid = sid
        self.frustum = self.create_frustum(grid_config['depth'],
                                           input_size, downsample)      # (D, fH, fW, 3)  3:(u, v, d)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.depth_net = nn.Conv2d(
            in_channels, self.D + self.out_channels, kernel_size=1, padding=0)
        self.accelerate = accelerate
        self.initial_flag = True
        self.collapse_z = collapse_z

    def create_grid_infos(self, x, y, z, **kwargs):
        """Generate the grid information including the lower bound, interval,
        and size.

        Args:
            x (tuple(float)): Config of grid alone x axis in format of
                (lower_bound, upper_bound, interval).
            y (tuple(float)): Config of grid alone y axis in format of
                (lower_bound, upper_bound, interval).
            z (tuple(float)): Config of grid alone z axis in format of
                (lower_bound, upper_bound, interval).
            **kwargs: Container for other potential parameters
        """
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [x, y, z]])     # (min_x, min_y, min_z)
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [x, y, z]])        # (dx, dy, dz)
        self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2]
                                       for cfg in [x, y, z]])                   # (Dx, Dy, Dz)

    def create_frustum(self, depth_cfg, input_size, downsample):
        """Generate the frustum template for each image.

        Args:
            depth_cfg (tuple(float)): Config of grid alone depth axis in format
                of (lower_bound, upper_bound, interval).
            input_size (tuple(int)): Size of input images in format of (height,
                width).
            downsample (int): Down sample scale factor from the input size to
                the feature size.
        Returns:
            frustum: (D, fH, fW, 3)  3:(u, v, d)
        """
        H_in, W_in = input_size
        H_feat, W_feat = H_in // downsample, W_in // downsample
        d = torch.arange(*depth_cfg, dtype=torch.float)\
            .view(-1, 1, 1).expand(-1, H_feat, W_feat)      # (D, fH, fW)
        self.D = d.shape[0]
        if self.sid:
            d_sid = torch.arange(self.D).float()
            depth_cfg_t = torch.tensor(depth_cfg).float()
            d_sid = torch.exp(torch.log(depth_cfg_t[0]) + d_sid / (self.D-1) *
                              torch.log((depth_cfg_t[1]-1) / depth_cfg_t[0]))
            d = d_sid.view(-1, 1, 1).expand(-1, H_feat, W_feat)

        x = torch.linspace(0, W_in - 1, W_feat,  dtype=torch.float)\
            .view(1, 1, W_feat).expand(self.D, H_feat, W_feat)      # (D, fH, fW)
        y = torch.linspace(0, H_in - 1, H_feat,  dtype=torch.float)\
            .view(1, H_feat, 1).expand(self.D, H_feat, W_feat)      # (D, fH, fW)

        return torch.stack((x, y, d), -1)    # (D, fH, fW, 3)  3:(u, v, d)

    def get_lidar_coor(self, sensor2ego, ego2global, cam2imgs, post_rots, post_trans,
                       bda):
        """Calculate the locations of the frustum points in the lidar
        coordinate system.

        Args:
            rots (torch.Tensor): Rotation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3, 3).
            trans (torch.Tensor): Translation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3).
            cam2imgs (torch.Tensor): Camera intrinsic matrixes in shape
                (B, N_cams, 3, 3).
            post_rots (torch.Tensor): Rotation in camera coordinate system in
                shape (B, N_cams, 3, 3). It is derived from the image view
                augmentation.
            post_trans (torch.Tensor): Translation in camera coordinate system
                derived from image view augmentation in shape (B, N_cams, 3).

        Returns:
            torch.tensor: Point coordinates in shape
                (B, N_cams, D, ownsample, 3)
        """
        B, N, _, _ = sensor2ego.shape

        # post-transformation
        # B x N x D x H x W x 3
        points = self.frustum.to(sensor2ego) - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3)\
            .matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        combine = sensor2ego[:,:,:3,:3].matmul(torch.inverse(cam2imgs))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += sensor2ego[:,:,:3, 3].view(B, N, 1, 1, 1, 3)
        points = bda.view(B, 1, 1, 1, 1, 3,
                          3).matmul(points.unsqueeze(-1)).squeeze(-1)
        return points

    def get_ego_coor(self, sensor2ego, ego2global, cam2imgs, post_rots, post_trans,
                     bda):
        """Calculate the locations of the frustum points in the lidar
        coordinate system.

        Args:
            sensor2ego (torch.Tensor): Transformation from camera coordinate system to
                ego coordinate system in shape (B, N_cams, 4, 4).
            ego2global (torch.Tensor): Translation from ego coordinate system to
                global coordinate system in shape (B, N_cams, 4, 4).
            cam2imgs (torch.Tensor): Camera intrinsic matrixes in shape
                (B, N_cams, 3, 3).
            post_rots (torch.Tensor): Rotation in camera coordinate system in
                shape (B, N_cams, 3, 3). It is derived from the image view
                augmentation.
            post_trans (torch.Tensor): Translation in camera coordinate system
                derived from image view augmentation in shape (B, N_cams, 3).
            bda (torch.Tensor): Transformation in bev. (B, 3, 3)

        Returns:
            torch.tensor: Point coordinates in shape (B, N, D, fH, fW, 3)
        """
        B, N, _, _ = sensor2ego.shape

        # post-transformation
        # (D, fH, fW, 3) - (B, N, 1, 1, 1, 3) --> (B, N, D, fH, fW, 3)
        points = self.frustum.to(sensor2ego) - post_trans.view(B, N, 1, 1, 1, 3)
        # (B, N, 1, 1, 1, 3, 3) @ (B, N, D, fH, fW, 3, 1)  --> (B, N, D, fH, fW, 3, 1)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3)\
            .matmul(points.unsqueeze(-1))

        # cam_to_ego
        # (B, N_, D, fH, fW, 3, 1)  3: (du, dv, d)
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        # R_{c->e} @ K^-1
        combine = sensor2ego[:, :, :3, :3].matmul(torch.inverse(cam2imgs))
        # (B, N, 1, 1, 1, 3, 3) @ (B, N, D, fH, fW, 3, 1)  --> (B, N, D, fH, fW, 3, 1)
        # --> (B, N, D, fH, fW, 3)
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        # (B, N, D, fH, fW, 3) + (B, N, 1, 1, 1, 3) --> (B, N, D, fH, fW, 3)
        points += sensor2ego[:, :, :3, 3].view(B, N, 1, 1, 1, 3)

        # (B, 1, 1, 1, 3, 3) @ (B, N, D, fH, fW, 3, 1) --> (B, N, D, fH, fW, 3, 1)
        # --> (B, N, D, fH, fW, 3)
        points = bda.view(B, 1, 1, 1, 1, 3,
                          3).matmul(points.unsqueeze(-1)).squeeze(-1)
        return points

    def init_acceleration_v2(self, coor):
        """Pre-compute the necessary information in acceleration including the
        index of points in the final feature.

        Args:
            coor (torch.tensor): Coordinate of points in lidar space in shape
                (B, N, D, H, W, 3).
            x (torch.tensor): Feature of points in shape
                (B, N_cams, D, H, W, C).
        """

        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)
        # ranks_bev: (N_points, ),
        # ranks_depth: (N_points, ),
        # ranks_feat: (N_points, ),
        # interval_starts: (N_pillar, )
        # interval_lengths: (N_pillar, )

        self.ranks_bev = ranks_bev.int().contiguous()
        self.ranks_feat = ranks_feat.int().contiguous()
        self.ranks_depth = ranks_depth.int().contiguous()
        self.interval_starts = interval_starts.int().contiguous()
        self.interval_lengths = interval_lengths.int().contiguous()

    def voxel_pooling_v2(self, coor, depth, feat):
        """
        Args:
            coor: (B, N, D, fH, fW, 3)
            depth: (B, N, D, fH, fW)
            feat: (B, N, C, fH, fW)
        Returns:
            bev_feat: (B, C*Dz(=1), Dy, Dx)
        """
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)
        # ranks_bev: (N_points, ),
        # ranks_depth: (N_points, ),
        # ranks_feat: (N_points, ),
        # interval_starts: (N_pillar, )
        # interval_lengths: (N_pillar, )
        if ranks_feat is None:
            print('warning ---> no points within the predefined '
                  'bev receptive field')
            dummy = torch.zeros(size=[
                feat.shape[0], feat.shape[2],
                int(self.grid_size[2]),
                int(self.grid_size[1]),
                int(self.grid_size[0])
            ]).to(feat)     # (B, C, Dz, Dy, Dx)
            dummy = torch.cat(dummy.unbind(dim=2), 1)   # (B, C*Dz, Dy, Dx)
            return dummy

        feat = feat.permute(0, 1, 3, 4, 2)      # (B, N, fH, fW, C)
        bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                          int(self.grid_size[1]), int(self.grid_size[0]),
                          feat.shape[-1])       # (B, Dz, Dy, Dx, C)
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape, interval_starts,
                               interval_lengths)    # (B, C, Dz, Dy, Dx)
        # collapse Z
        if self.collapse_z:
            bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)     # (B, C*Dz, Dy, Dx)
        return bev_feat

    def voxel_pooling_prepare_v2(self, coor):
        """Data preparation for voxel pooling.
        Args:
            coor (torch.tensor): Coordinate of points in the lidar space in
                shape (B, N, D, H, W, 3).
        Returns:
            tuple[torch.tensor]:
                ranks_bev: Rank of the voxel that a point is belong to in shape (N_points, ),
                    rank介于(0, B*Dx*Dy*Dz-1).
                ranks_depth: Reserved index of points in the depth space in shape (N_Points),
                    rank介于(0, B*N*D*fH*fW-1).
                ranks_feat: Reserved index of points in the feature space in shape (N_Points),
                    rank介于(0, B*N*fH*fW-1).
                interval_starts: (N_pillar, )
                interval_lengths: (N_pillar, )
        """
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        # record the index of selected points for acceleration purpose
        ranks_depth = torch.range(
            0, num_points - 1, dtype=torch.int, device=coor.device)    # (B*N*D*H*W, ), [0, 1, ..., B*N*D*fH*fW-1]
        ranks_feat = torch.range(
            0, num_points // D - 1, dtype=torch.int, device=coor.device)   # [0, 1, ...,B*N*fH*fW-1]
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()     # (B*N*D*fH*fW, )

        # convert coordinate into the voxel space
        # ((B, N, D, fH, fW, 3) - (3, )) / (3, ) --> (B, N, D, fH, fW, 3)   3:(x, y, z)  grid coords.
        coor = ((coor - self.grid_lower_bound.to(coor)) /
                self.grid_interval.to(coor))
        coor = coor.long().view(num_points, 3)      # (B, N, D, fH, fW, 3) --> (B*N*D*fH*fW, 3)
        # (B, N*D*fH*fW) --> (B*N*D*fH*fW, 1)
        batch_idx = torch.range(0, B - 1).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)      # (B*N*D*fH*fW, 4)   4: (x, y, z, batch_id)

        # filter out points that are outside box
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])
        if len(kept) == 0:
            return None, None, None, None, None

        # (N_points, 4), (N_points, ), (N_points, )
        coor, ranks_depth, ranks_feat = \
            coor[kept], ranks_depth[kept], ranks_feat[kept]

        # get tensors from the same voxel next to each other
        ranks_bev = coor[:, 3] * (
            self.grid_size[2] * self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 1] * self.grid_size[0] + coor[:, 0]
        order = ranks_bev.argsort()
        # (N_points, ), (N_points, ), (N_points, )
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
        ), ranks_feat.int().contiguous(), interval_starts.int().contiguous(
        ), interval_lengths.int().contiguous()

    def pre_compute(self, input):
        if self.initial_flag:
            coor = self.get_ego_coor(*input[1:7])       # (B, N, D, fH, fW, 3)
            self.init_acceleration_v2(coor)
            self.initial_flag = False

    def view_transform_core(self, input, depth, tran_feat):
        """
        Args:
            input (list(torch.tensor)):
                imgs:  (B, N, 3, H, W)        # N_views = 6 * (N_history + 1)
                sensor2egos: (B, N, 4, 4)
                ego2globals: (B, N, 4, 4)
                intrins:     (B, N, 3, 3)
                post_rots:   (B, N, 3, 3)
                post_trans:  (B, N, 3)
                bda_rot:  (B, 3, 3)
            depth:  (B*N, D, fH, fW)
            tran_feat: (B*N, C, fH, fW)
        Returns:
            bev_feat: (B, C*Dz(=1), Dy, Dx)
            depth: (B*N, D, fH, fW)
        """
        B, N, C, H, W = input[0].shape

        # Lift-Splat
        if self.accelerate:
            feat = tran_feat.view(B, N, self.out_channels, H, W)      # (B, N, C, fH, fW)
            feat = feat.permute(0, 1, 3, 4, 2)      # (B, N, fH, fW, C)
            depth = depth.view(B, N, self.D, H, W)      # (B, N, D, fH, fW)
            bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                              int(self.grid_size[1]), int(self.grid_size[0]),
                              feat.shape[-1])   # (B, Dz, Dy, Dx, C)
            bev_feat = bev_pool_v2(depth, feat, self.ranks_depth,
                                   self.ranks_feat, self.ranks_bev,
                                   bev_feat_shape, self.interval_starts,
                                   self.interval_lengths)   # (B, C, Dz, Dy, Dx)

            bev_feat = bev_feat.squeeze(2)      # (B, C, Dy, Dx)
        else:
            coor = self.get_ego_coor(*input[1:7])   # (B, N, D, fH, fW, 3)
            bev_feat = self.voxel_pooling_v2(
                coor, depth.view(B, N, self.D, H, W),
                tran_feat.view(B, N, self.out_channels, H, W))      # (B, C*Dz(=1), Dy, Dx)
        return bev_feat, depth

    def view_transform(self, input, depth, tran_feat):
        """
        Args:
            input (list(torch.tensor)):
                imgs:  (B, N, C, H, W)        # N_views = 6 * (N_history + 1)
                sensor2egos: (B, N, 4, 4)
                ego2globals: (B, N, 4, 4)
                intrins:     (B, N, 3, 3)
                post_rots:   (B, N, 3, 3)
                post_trans:  (B, N, 3)
                bda_rot:  (B, 3, 3)
            depth:  (B*N, D, fH, fW)
            tran_feat: (B*N, C, fH, fW)
        Returns:
            bev_feat: (B, C, Dy, Dx)
            depth: (B*N, D, fH, fW)
        """
        if self.accelerate:
            self.pre_compute(input)
        return self.view_transform_core(input, depth, tran_feat)

    def forward(self, input):
        """Transform image-view feature into bird-eye-view feature.

        Args:
            input (list(torch.tensor)):
                imgs:  (B, N_views, 3, H, W)        # N_views = 6 * (N_history + 1)
                sensor2egos: (B, N_views, 4, 4)
                ego2globals: (B, N_views, 4, 4)
                intrins:     (B, N_views, 3, 3)
                post_rots:   (B, N_views, 3, 3)
                post_trans:  (B, N_views, 3)
                bda_rot:  (B, 3, 3)
        Returns:
            bev_feat: (B, C, Dy, Dx)
            depth: (B*N, D, fH, fW)
        """
        x = input[0]    # (B, N, C_in, fH, fW)
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)      # (B*N, C_in, fH, fW)

        # (B*N, C_in, fH, fW) --> (B*N, D+C, fH, fW)
        x = self.depth_net(x)
        depth_digit = x[:, :self.D, ...]    # (B*N, D, fH, fW)
        tran_feat = x[:, self.D:self.D + self.out_channels, ...]    # (B*N, C, fH, fW)
        depth = depth_digit.softmax(dim=1)
        return self.view_transform(input, depth, tran_feat)

    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda):
        return None


@NECKS.register_module()
class LSSViewTransformerBEVDepth(LSSViewTransformer):
    def __init__(self, loss_depth_weight=3.0, depthnet_cfg=dict(), **kwargs):
        super(LSSViewTransformerBEVDepth, self).__init__(**kwargs)
        self.loss_depth_weight = loss_depth_weight
        self.depth_net = DepthNet(
            in_channels=self.in_channels,
            mid_channels=self.in_channels,
            context_channels=self.out_channels,
            depth_channels=self.D,
            **depthnet_cfg)

    def get_mlp_input(self, sensor2ego, ego2global, intrin, post_rot, post_tran, bda):
        """
        Args:
            sensor2ego: (B, N_views=6, 4, 4)
            ego2global: (B, N_views=6, 4, 4)
            intrin: (B, N_views, 3, 3)
            post_rot: (B, N_views, 3, 3)
            post_tran: (B, N_views, 3)
            bda: (B, 3, 3)
        Returns:
            mlp_input: (B, N_views, 27)
        """
        B, N, _, _ = sensor2ego.shape
        bda = bda.view(B, 1, 3, 3).repeat(1, N, 1, 1)   # (B, 3, 3) --> (B, N, 3, 3)
        mlp_input = torch.stack([
            intrin[:, :, 0, 0],     # fx
            intrin[:, :, 1, 1],     # fy
            intrin[:, :, 0, 2],     # cx
            intrin[:, :, 1, 2],     # cy
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
            bda[:, :, 0, 0],
            bda[:, :, 0, 1],
            bda[:, :, 1, 0],
            bda[:, :, 1, 1],
            bda[:, :, 2, 2]
        ], dim=-1)      # (B, N_views, 15)
        sensor2ego = sensor2ego[:, :, :3, :].reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)      # (B, N_views, 27)
        return mlp_input

    def forward(self, input, stereo_metas=None):
        """
        Args:
            input (list(torch.tensor)):
                imgs:  (B, N_views, 3, H, W)        # N_views = 6 * (N_history + 1)
                sensor2egos: (B, N_views, 4, 4)
                ego2globals: (B, N_views, 4, 4)
                intrins:     (B, N_views, 3, 3)
                post_rots:   (B, N_views, 3, 3)
                post_trans:  (B, N_views, 3)
                bda_rot:  (B, 3, 3)
                mlp_input: (B, N_views, 27)
            stereo_metas:  None or dict{
                k2s_sensor: (B, N_views, 4, 4)
                intrins: (B, N_views, 3, 3)
                post_rots: (B, N_views, 3, 3)
                post_trans: (B, N_views, 3)
                frustum: (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
                cv_downsample: 4,
                downsample: self.img_view_transformer.downsample=16,
                grid_config: self.img_view_transformer.grid_config,
                cv_feat_list: [feat_prev_iv, stereo_feat]
            }
        Returns:
            bev_feat: (B, C, Dy, Dx)
            depth: (B*N, D, fH, fW)
        """
        (x, rots, trans, intrins, post_rots, post_trans, bda,
         mlp_input) = input[:8]

        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)      # (B*N_views, C, fH, fW)
        x = self.depth_net(x, mlp_input, stereo_metas)      # (B*N_views, D+C_context, fH, fW)
        depth_digit = x[:, :self.D, ...]    # (B*N_views, D, fH, fW)
        tran_feat = x[:, self.D:self.D + self.out_channels, ...]    # (B*N_views, C_context, fH, fW)
        depth = depth_digit.softmax(dim=1)  # (B*N_views, D, fH, fW)
        bev_feat, depth = self.view_transform(input, depth, tran_feat)
        return bev_feat, depth

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: (B, N_views, img_h, img_w)
        Output:
            gt_depths: (B*N_views*fH*fW, D)
        """
        B, N, H, W = gt_depths.shape
        # (B*N_views, fH, downsample, fW, downsample, 1)
        gt_depths = gt_depths.view(B * N,
                                   H // self.downsample, self.downsample,
                                   W // self.downsample, self.downsample,
                                   1)
        # (B*N_views, fH, fW, 1, downsample, downsample)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        # (B*N_views*fH*fW, downsample, downsample)
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        # (B*N_views, fH, fW)
        gt_depths = gt_depths.view(B * N, H // self.downsample, W // self.downsample)

        if not self.sid:
            # (D - (min_dist - interval_dist)) / interval_dist
            # = (D - min_dist) / interval_dist + 1
            gt_depths = (gt_depths - (self.grid_config['depth'][0] -
                                      self.grid_config['depth'][2])) / \
                        self.grid_config['depth'][2]
        else:
            gt_depths = torch.log(gt_depths) - torch.log(
                torch.tensor(self.grid_config['depth'][0]).float())
            gt_depths = gt_depths * (self.D - 1) / torch.log(
                torch.tensor(self.grid_config['depth'][1] - 1.).float() /
                self.grid_config['depth'][0])
            gt_depths = gt_depths + 1.

        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))     # (B*N_views, fH, fW)
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:]   # (B*N_views*fH*fW, D)
        return gt_depths.float()

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        """
        Args:
            depth_labels: (B, N_views, img_h, img_w)
            depth_preds: (B*N_views, D, fH, fW)
        Returns:

        """
        depth_labels = self.get_downsampled_gt_depth(depth_labels)      # (B*N_views*fH*fW, D)
        # (B*N_views, D, fH, fW) --> (B*N_views, fH, fW, D) --> (B*N_views*fH*fW, D)
        depth_preds = depth_preds.permute(0, 2, 3,
                                          1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        return self.loss_depth_weight * depth_loss


@NECKS.register_module()
class LSSViewTransformerBEVStereo(LSSViewTransformerBEVDepth):
    def __init__(self,  **kwargs):
        super(LSSViewTransformerBEVStereo, self).__init__(**kwargs)
        # (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
        self.cv_frustum = self.create_frustum(kwargs['grid_config']['depth'],
                                              kwargs['input_size'],
                                              downsample=4)
