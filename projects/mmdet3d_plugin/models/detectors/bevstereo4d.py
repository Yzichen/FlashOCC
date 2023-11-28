# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32

from mmdet3d.models import DETECTORS
from mmdet3d.models import builder
from .bevdepth4d import BEVDepth4D
from mmdet.models.backbones.resnet import ResNet


@DETECTORS.register_module()
class BEVStereo4D(BEVDepth4D):
    def __init__(self, **kwargs):
        super(BEVStereo4D, self).__init__(**kwargs)
        self.extra_ref_frames = 1
        self.temporal_frame = self.num_frame
        self.num_frame += self.extra_ref_frames

    def extract_stereo_ref_feat(self, x):
        """
        Args:
            x: (B, N_views, 3, H, W)
        Returns:
            x: (B*N_views, C_stereo, fH_stereo, fW_stereo)
        """
        B, N, C, imH, imW = x.shape
        x = x.view(B * N, C, imH, imW)      # (B*N_views, 3, H, W)
        if isinstance(self.img_backbone, ResNet):
            if self.img_backbone.deep_stem:
                x = self.img_backbone.stem(x)
            else:
                x = self.img_backbone.conv1(x)
                x = self.img_backbone.norm1(x)
                x = self.img_backbone.relu(x)
            x = self.img_backbone.maxpool(x)
            for i, layer_name in enumerate(self.img_backbone.res_layers):
                res_layer = getattr(self.img_backbone, layer_name)
                x = res_layer(x)
                return x
        else:
            x = self.img_backbone.patch_embed(x)
            hw_shape = (self.img_backbone.patch_embed.DH,
                        self.img_backbone.patch_embed.DW)
            if self.img_backbone.use_abs_pos_embed:
                x = x + self.img_backbone.absolute_pos_embed
            x = self.img_backbone.drop_after_pos(x)

            for i, stage in enumerate(self.img_backbone.stages):
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
                out = out.view(-1,  *out_hw_shape,
                               self.img_backbone.num_features[i])
                out = out.permute(0, 3, 1, 2).contiguous()
                return out

    def prepare_bev_feat(self, img, sensor2keyego, ego2global, intrin,
                         post_rot, post_tran, bda, mlp_input, feat_prev_iv,
                         k2s_sensor, extra_ref_frame):
        """
        Args:
            img:  (B, N_views, 3, H, W)
            sensor2keyego: (B, N_views, 4, 4)
            ego2global: (B, N_views, 4, 4)
            intrin: (B, N_views, 3, 3)
            post_rot: (B, N_views, 3, 3)
            post_tran: (B, N_views, 3)
            bda: (B, 3, 3)
            mlp_input: (B, N_views, 27)
            feat_prev_iv: (B*N_views, C_stereo, fH_stereo, fW_stereo) or None
            k2s_sensor: (B, N_views, 4, 4) or None
            extra_ref_frame:
        Returns:
            bev_feat: (B, C, Dy, Dx)
            depth: (B*N, D, fH, fW)
            stereo_feat: (B*N_views, C_stereo, fH_stereo, fW_stereo)
        """
        if extra_ref_frame:
            stereo_feat = self.extract_stereo_ref_feat(img)     # (B*N_views, C_stereo, fH_stereo, fW_stereo)
            return None, None, stereo_feat
        # x: (B, N_views, C, fH, fW)
        # stereo_feat: (B*N, C_stereo, fH_stereo, fW_stereo)
        x, stereo_feat = self.image_encoder(img, stereo=True)

        # 建立cost volume 所需的信息.
        metas = dict(k2s_sensor=k2s_sensor,     # (B, N_views, 4, 4)
                     intrins=intrin,            # (B, N_views, 3, 3)
                     post_rots=post_rot,        # (B, N_views, 3, 3)
                     post_trans=post_tran,      # (B, N_views, 3)
                     frustum=self.img_view_transformer.cv_frustum.to(x),    # (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
                     cv_downsample=4,
                     downsample=self.img_view_transformer.downsample,
                     grid_config=self.img_view_transformer.grid_config,
                     cv_feat_list=[feat_prev_iv, stereo_feat]
                     )
        # bev_feat: (B, C * Dz(=1), Dy, Dx)
        # depth: (B * N, D, fH, fW)
        bev_feat, depth = self.img_view_transformer(
            [x, sensor2keyego, ego2global, intrin, post_rot, post_tran, bda,
             mlp_input], metas)
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]    # (B, C, Dy, Dx)
        return bev_feat, depth, stereo_feat

    def extract_img_feat_sequential(self, inputs, feat_prev):
        """
        Args:
            inputs:
                curr_img: (1, N_views, 3, H, W)
                sensor2keyegos_curr:  (N_prev, N_views, 4, 4)
                ego2globals_curr:  (N_prev, N_views, 4, 4)
                intrins:  (1, N_views, 3, 3)
                sensor2keyegos_prev:  (N_prev, N_views, 4, 4)
                ego2globals_prev:  (N_prev, N_views, 4, 4)
                post_rots:  (1, N_views, 3, 3)
                post_trans: (1, N_views, 3, )
                bda_curr:  (N_prev, 3, 3)
                feat_prev_iv:
                curr2adjsensor: (1, N_views, 4, 4)
            feat_prev: (N_prev, C, Dy, Dx)
        Returns:

        """
        imgs, sensor2keyegos_curr, ego2globals_curr, intrins = inputs[:4]
        sensor2keyegos_prev, _, post_rots, post_trans, bda = inputs[4:9]
        feat_prev_iv, curr2adjsensor = inputs[9:]

        bev_feat_list = []
        mlp_input = self.img_view_transformer.get_mlp_input(
            sensor2keyegos_curr[0:1, ...], ego2globals_curr[0:1, ...],
            intrins, post_rots, post_trans, bda[0:1, ...])
        inputs_curr = (imgs, sensor2keyegos_curr[0:1, ...],
                       ego2globals_curr[0:1, ...], intrins, post_rots,
                       post_trans, bda[0:1, ...], mlp_input, feat_prev_iv,
                       curr2adjsensor, False)

        # (1, C, Dx, Dy), (1*N, D, fH, fW)
        bev_feat, depth, _ = self.prepare_bev_feat(*inputs_curr)
        bev_feat_list.append(bev_feat)

        # align the feat_prev
        _, C, H, W = feat_prev.shape
        # feat_prev: (N_prev, C, Dy, Dx)
        feat_prev = \
            self.shift_feature(feat_prev,   # (N_prev, C, Dy, Dx)
                               [sensor2keyegos_curr,    # (N_prev, N_views, 4, 4)
                                sensor2keyegos_prev],   # (N_prev, N_views, 4, 4)
                               bda  # (N_prev, 3, 3)
                               )
        bev_feat_list.append(feat_prev.view(1, (self.num_frame - 2) * C, H, W))     # (1, N_prev*C, Dy, Dx)

        bev_feat = torch.cat(bev_feat_list, dim=1)      # (1, N_frames*C, Dy, Dx)
        x = self.bev_encoder(bev_feat)
        return [x], depth

    def extract_img_feat(self,
                         img_inputs,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        """
        Args:
            img_inputs:
                imgs:  (B, N, 3, H, W)        # N = 6 * (N_history + 1)
                sensor2egos: (B, N, 4, 4)
                ego2globals: (B, N, 4, 4)
                intrins:     (B, N, 3, 3)
                post_rots:   (B, N, 3, 3)
                post_trans:  (B, N, 3)
                bda_rot:  (B, 3, 3)
            img_metas:
            **kwargs:
        Returns:
            x: [(B, C', H', W'), ]
            depth: (B*N_views, D, fH, fW)
        """
        if sequential:
            return self.extract_img_feat_sequential(img_inputs, kwargs['feat_prev'])

        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, curr2adjsensor = self.prepare_inputs(img_inputs, stereo=True)

        """Extract features of images."""
        bev_feat_list = []
        depth_key_frame = None
        feat_prev_iv = None
        for fid in range(self.num_frame-1, -1, -1):
            img, sensor2keyego, ego2global, intrin, post_rot, post_tran = \
                imgs[fid], sensor2keyegos[fid], ego2globals[fid], intrins[fid], \
                post_rots[fid], post_trans[fid]
            key_frame = fid == 0
            extra_ref_frame = fid == self.num_frame-self.extra_ref_frames
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin,
                    post_rot, post_tran, bda)   # (B, N_views, 27)

                inputs_curr = (img, sensor2keyego, ego2global, intrin,
                               post_rot, post_tran, bda, mlp_input,
                               feat_prev_iv, curr2adjsensor[fid],
                               extra_ref_frame)

                if key_frame:
                    bev_feat, depth, feat_curr_iv = \
                        self.prepare_bev_feat(*inputs_curr)
                    depth_key_frame = depth
                else:
                    with torch.no_grad():
                        bev_feat, depth, feat_curr_iv = \
                            self.prepare_bev_feat(*inputs_curr)

                if not extra_ref_frame:
                    bev_feat_list.append(bev_feat)

                if not key_frame:
                    feat_prev_iv = feat_curr_iv

        if pred_prev:
            assert self.align_after_view_transfromation
            assert sensor2keyegos[0].shape[0] == 1      # batch_size = 1
            feat_prev = torch.cat(bev_feat_list[1:], dim=0)

            # (1, N_views, 4, 4) --> (N_prev, N_views, 4, 4)
            ego2globals_curr = \
                ego2globals[0].repeat(self.num_frame - 2, 1, 1, 1)
            # (1, N_views, 4, 4) --> (N_prev, N_views, 4, 4)
            sensor2keyegos_curr = \
                sensor2keyegos[0].repeat(self.num_frame - 2, 1, 1, 1)
            ego2globals_prev = torch.cat(ego2globals[1:-1], dim=0)            # (N_prev, N_views, 4, 4)
            sensor2keyegos_prev = torch.cat(sensor2keyegos[1:-1], dim=0)      # (N_prev, N_views, 4, 4)
            bda_curr = bda.repeat(self.num_frame - 2, 1, 1)     # (N_prev, 3, 3)

            return feat_prev, [imgs[0],     # (1, N_views, 3, H, W)
                               sensor2keyegos_curr,     # (N_prev, N_views, 4, 4)
                               ego2globals_curr,        # (N_prev, N_views, 4, 4)
                               intrins[0],          # (1, N_views, 3, 3)
                               sensor2keyegos_prev,     # (N_prev, N_views, 4, 4)
                               ego2globals_prev,        # (N_prev, N_views, 4, 4)
                               post_rots[0],    # (1, N_views, 3, 3)
                               post_trans[0],   # (1, N_views, 3, )
                               bda_curr,       # (N_prev, 3, 3)
                               feat_prev_iv,
                               curr2adjsensor[0]]

        if not self.with_prev:
            bev_feat_key = bev_feat_list[0]
            if len(bev_feat_key.shape) == 4:
                b, c, h, w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1),
                                  h, w]).to(bev_feat_key), bev_feat_key]
            else:
                b, c, z, h, w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1), z,
                                  h, w]).to(bev_feat_key), bev_feat_key]

        if self.align_after_view_transfromation:
            for adj_id in range(self.num_frame-2):
                bev_feat_list[adj_id] = self.shift_feature(
                    bev_feat_list[adj_id],  # (B, C, Dy, Dx)
                    [sensor2keyegos[0],     # (B, N_views, 4, 4)
                     sensor2keyegos[self.num_frame-2-adj_id]],  # (B, N_views, 4, 4)
                    bda  # (B, 3, 3)
                )   # (B, C, Dy, Dx)

        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x], depth_key_frame












