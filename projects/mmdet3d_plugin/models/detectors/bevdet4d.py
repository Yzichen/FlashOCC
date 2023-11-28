# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32

from mmdet3d.models import DETECTORS
from mmdet3d.models import builder
from .bevdet import BEVDet


@DETECTORS.register_module()
class BEVDet4D(BEVDet):
    r"""BEVDet4D paradigm for multi-camera 3D object detection.

    Please refer to the `paper <https://arxiv.org/abs/2203.17054>`_

    Args:
        pre_process (dict | None): Configuration dict of BEV pre-process net.
        align_after_view_transfromation (bool): Whether to align the BEV
            Feature after view transformation. By default, the BEV feature of
            the previous frame is aligned during the view transformation.
        num_adj (int): Number of adjacent frames.
        with_prev (bool): Whether to set the BEV feature of previous frame as
            all zero. By default, False.
    """
    def __init__(self,
                 pre_process=None,
                 align_after_view_transfromation=False,
                 num_adj=1,
                 with_prev=True,
                 **kwargs):
        super(BEVDet4D, self).__init__(**kwargs)
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process)

        self.align_after_view_transfromation = align_after_view_transfromation
        self.num_frame = num_adj + 1

        self.with_prev = with_prev
        self.grid = None

    def gen_grid(self, input, sensor2keyegos, bda, bda_adj=None):
        """
        Args:
            input: (B, C, Dy, Dx)  bev_feat
            sensor2keyegos: List[
                curr_sensor-->key_ego: (B, N_views, 4, 4)
                prev_sensor-->key_ego: (B, N_views, 4, 4)
            ]
            bda:  (B, 3, 3)
            bda_adj: None
        Returns:
            grid: (B, Dy, Dx, 2)
        """
        B, C, H, W = input.shape
        v = sensor2keyegos[0].shape[0]  # N_views
        if self.grid is None:
            # generate grid
            xs = torch.linspace(
                0, W - 1, W, dtype=input.dtype,
                device=input.device).view(1, W).expand(H, W)    # (Dy, Dx)
            ys = torch.linspace(
                0, H - 1, H, dtype=input.dtype,
                device=input.device).view(H, 1).expand(H, W)    # (Dy, Dx)
            grid = torch.stack((xs, ys, torch.ones_like(xs)), -1)   # (Dy, Dx, 3)   3: (x, y, 1)
            self.grid = grid
        else:
            grid = self.grid
        # (Dy, Dx, 3)  --> (1, Dy, Dx, 3) --> (B, Dy, Dx, 3) --> (B, Dy, Dx, 3, 1))     3: (grid_x, grid_y, 1)
        grid = grid.view(1, H, W, 3).expand(B, H, W, 3).view(B, H, W, 3, 1)

        curr_sensor2keyego = sensor2keyegos[0][:, 0:1, :, :]    # (B, 1, 4, 4)
        prev_sensor2keyego = sensor2keyegos[1][:, 0:1, :, :]    # (B, 1, 4, 4)

        # add bev data augmentation
        bda_ = torch.zeros((B, 1, 4, 4), dtype=grid.dtype).to(grid)     # (B, 1, 4, 4)
        bda_[:, :, :3, :3] = bda.unsqueeze(1)
        bda_[:, :, 3, 3] = 1
        curr_sensor2keyego = bda_.matmul(curr_sensor2keyego)        # (B, 1, 4, 4)
        if bda_adj is not None:
            bda_ = torch.zeros((B, 1, 4, 4), dtype=grid.dtype).to(grid)
            bda_[:, :, :3, :3] = bda_adj.unsqueeze(1)
            bda_[:, :, 3, 3] = 1
        prev_sensor2keyego = bda_.matmul(prev_sensor2keyego)        # (B, 1, 4, 4)

        # transformation from current ego frame to adjacent ego frame
        # key_ego --> prev_cam_front --> prev_ego
        keyego2adjego = curr_sensor2keyego.matmul(torch.inverse(prev_sensor2keyego))
        keyego2adjego = keyego2adjego.unsqueeze(dim=1)      # (B, 1, 1, 4, 4)

        # (B, 1, 1, 3, 3)
        keyego2adjego = keyego2adjego[..., [True, True, False, True], :][..., [True, True, False, True]]

        # x = grid_x * vx + x_min;  y = grid_y * vy + y_min;
        # feat2bev:
        # [[vx, 0, x_min],
        #  [0, vy, y_min],
        #  [0,  0,   1  ]]
        feat2bev = torch.zeros((3, 3), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.img_view_transformer.grid_interval[0]
        feat2bev[1, 1] = self.img_view_transformer.grid_interval[1]
        feat2bev[0, 2] = self.img_view_transformer.grid_lower_bound[0]
        feat2bev[1, 2] = self.img_view_transformer.grid_lower_bound[1]
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.view(1, 3, 3)       # (1, 3, 3)

        # curr_feat_grid --> key ego --> prev_cam --> prev_ego --> prev_feat_grid
        tf = torch.inverse(feat2bev).matmul(keyego2adjego).matmul(feat2bev)    # (B, 1, 1, 3, 3)
        grid = tf.matmul(grid)      # (B, Dy, Dx, 3, 1)    3: (grid_x, grid_y, 1)
        normalize_factor = torch.tensor([W - 1.0, H - 1.0],
                                        dtype=input.dtype,
                                        device=input.device)    # (2, )
        # (B, Dy, Dx, 2)
        grid = grid[:, :, :, :2, 0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0
        return grid

    @force_fp32()
    def shift_feature(self, input, sensor2keyegos, bda, bda_adj=None):
        """
        Args:
            input: (B, C, Dy, Dx)  bev_feat
            sensor2keyegos: List[
                curr_sensor-->key_ego: (B, N_views, 4, 4)
                prev_sensor-->key_ego: (B, N_views, 4, 4)
            ]
            bda:  (B, 3, 3)
            bda_adj: None
        Returns:
            output: aligned bev feat (B, C, Dy, Dx).
        """
        grid = self.gen_grid(input, sensor2keyegos, bda, bda_adj=bda_adj)   # grid: (B, Dy, Dx, 2),  介于(-1, 1)
        output = F.grid_sample(input, grid.to(input.dtype), align_corners=True)     # (B, C, Dy, Dx)
        return output

    def prepare_bev_feat(self, img, sensor2egos, ego2globals, intrin, post_rot, post_tran,
                         bda, mlp_input):
        """
        Args:
            imgs:  (B, N_views, 3, H, W)
            sensor2egos: (B, N_views, 4, 4)
            ego2globals: (B, N_views, 4, 4)
            intrins:     (B, N_views, 3, 3)
            post_rots:   (B, N_views, 3, 3)
            post_trans:  (B, N_views, 3)
            bda_rot:  (B, 3, 3)
            mlp_input:
        Returns:
            bev_feat: (B, C, Dy, Dx)
            depth: (B*N, D, fH, fW)
        """
        x, _ = self.image_encoder(img)      # x: (B, N, C, fH, fW)
        # bev_feat: (B, C * Dz(=1), Dy, Dx)
        # depth: (B * N, D, fH, fW)
        bev_feat, depth = self.img_view_transformer(
            [x, sensor2egos, ego2globals, intrin, post_rot, post_tran, bda, mlp_input])

        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]    # (B, C, Dy, Dx)
        return bev_feat, depth

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
            feat_prev: (N_prev, C, Dy, Dx)
        Returns:

        """
        imgs, sensor2keyegos_curr, ego2globals_curr, intrins = inputs[:4]
        sensor2keyegos_prev, _, post_rots, post_trans, bda = inputs[4:]
        bev_feat_list = []
        mlp_input = self.img_view_transformer.get_mlp_input(
            sensor2keyegos_curr[0:1, ...], ego2globals_curr[0:1, ...],
            intrins, post_rots, post_trans, bda[0:1, ...])
        inputs_curr = (imgs, sensor2keyegos_curr[0:1, ...],
                       ego2globals_curr[0:1, ...], intrins, post_rots,
                       post_trans, bda[0:1, ...], mlp_input)

        # (1, C, Dx, Dy), (1*N, D, fH, fW)
        bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
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
        bev_feat_list.append(feat_prev.view(1, (self.num_frame - 1) * C, H, W))     # (1, N_prev*C, Dy, Dx)

        bev_feat = torch.cat(bev_feat_list, dim=1)      # (1, N_frames*C, Dy, Dx)
        x = self.bev_encoder(bev_feat)
        return [x], depth

    def prepare_inputs(self, img_inputs, stereo=False):
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
            stereo: bool

        Returns:
            imgs: List[(B, N_views, C, H, W), (B, N_views, C, H, W), ...]       len = N_frames
            sensor2keyegos: List[(B, N_views, 4, 4), (B, N_views, 4, 4), ...]
            ego2globals: List[(B, N_views, 4, 4), (B, N_views, 4, 4), ...]
            intrins: List[(B, N_views, 3, 3), (B, N_views, 3, 3), ...]
            post_rots: List[(B, N_views, 3, 3), (B, N_views, 3, 3), ...]
            post_trans: List[(B, N_views, 3), (B, N_views, 3), ...]
            bda: (B, 3, 3)
        """
        B, N, C, H, W = img_inputs[0].shape
        N = N // self.num_frame     # N_views = 6
        imgs = img_inputs[0].view(B, N, self.num_frame, C, H, W)    # (B, N_views, N_frames, C, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]     # List[(B, N_views, C, H, W), (B, N_views, C, H, W), ...]
        sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            img_inputs[1:7]

        sensor2egos = sensor2egos.view(B, self.num_frame, N, 4, 4)
        ego2globals = ego2globals.view(B, self.num_frame, N, 4, 4)

        # calculate the transformation from sensor to key ego
        # key_ego --> global  (B, 1, 1, 4, 4)
        keyego2global = ego2globals[:, 0, 0, ...].unsqueeze(1).unsqueeze(1)
        # global --> key_ego  (B, 1, 1, 4, 4)
        global2keyego = torch.inverse(keyego2global.double())
        # sensor --> ego --> global --> key_ego
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()     # (B, N_frames, N_views, 4, 4)
        sensor2keyegos = sensor2keyegos.float()

        # --------------------  for stereo --------------------------
        curr2adjsensor = None
        if stereo:
            # (B, N_frames, N_views, 4, 4),  (B, N_frames, N_views, 4, 4)
            sensor2egos_cv, ego2globals_cv = sensor2egos, ego2globals
            sensor2egos_curr = \
                sensor2egos_cv[:, :self.temporal_frame, ...].double()   # (B, N_temporal=2, N_views, 4, 4)
            ego2globals_curr = \
                ego2globals_cv[:, :self.temporal_frame, ...].double()   # (B, N_temporal=2, N_views, 4, 4)
            sensor2egos_adj = \
                sensor2egos_cv[:, 1:self.temporal_frame + 1, ...].double()    # (B, N_temporal=2, N_views, 4, 4)
            ego2globals_adj = \
                ego2globals_cv[:, 1:self.temporal_frame + 1, ...].double()    # (B, N_temporal=2, N_views, 4, 4)

            # curr_sensor --> curr_ego --> global --> prev_ego --> prev_sensor
            curr2adjsensor = \
                torch.inverse(ego2globals_adj @ sensor2egos_adj) \
                @ ego2globals_curr @ sensor2egos_curr       # (B, N_temporal=2, N_views, 4, 4)
            curr2adjsensor = curr2adjsensor.float()         # (B, N_temporal=2, N_views, 4, 4)
            curr2adjsensor = torch.split(curr2adjsensor, 1, 1)
            curr2adjsensor = [p.squeeze(1) for p in curr2adjsensor]
            curr2adjsensor.extend([None for _ in range(self.extra_ref_frames)])
            # curr2adjsensor: List[(B, N_views, 4, 4), (B, N_views, 4, 4), None]
            assert len(curr2adjsensor) == self.num_frame
        # --------------------  for stereo --------------------------

        extra = [
            sensor2keyegos,     # (B, N_frames, N_views, 4, 4)
            ego2globals,        # (B, N_frames, N_views, 4, 4)
            intrins.view(B, self.num_frame, N, 3, 3),   # (B, N_frames, N_views, 3, 3)
            post_rots.view(B, self.num_frame, N, 3, 3),     # (B, N_frames, N_views, 3, 3)
            post_trans.view(B, self.num_frame, N, 3)        # (B, N_frames, N_views, 3)
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        sensor2keyegos, ego2globals, intrins, post_rots, post_trans = extra
        return imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
               bda, curr2adjsensor

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
        bda, _ = self.prepare_inputs(img_inputs)

        """Extract features of images."""
        bev_feat_list = []
        depth_list = []
        key_frame = True  # back propagation for key frame only
        for img, sensor2keyego, ego2global, intrin, post_rot, post_tran in zip(
                imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans):
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]

                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin, post_rot, post_tran, bda)    # (B, N_views, 27)

                inputs_curr = (img, sensor2keyego, ego2global, intrin, post_rot,
                               post_tran, bda, mlp_input)
                if key_frame:
                    # bev_feat: (B, C, Dy, Dx)
                    # depth: (B*N_views, D, fH, fW)
                    bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
                else:
                    with torch.no_grad():
                        bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
            else:
                # https://github.com/HuangJunJie2017/BEVDet/issues/275
                bev_feat = torch.zeros_like(bev_feat_list[0])
                depth = None
            bev_feat_list.append(bev_feat)
            depth_list.append(depth)
            key_frame = False

        # bev_feat_list: List[(B, C, Dy, Dx), (B, C, Dy, Dx), ...]
        # depth_list: List[(B*N_views, D, fH, fW), (B*N_views, D, fH, fW), ...]

        if pred_prev:
            assert self.align_after_view_transfromation
            assert sensor2keyegos[0].shape[0] == 1      # batch_size = 1
            feat_prev = torch.cat(bev_feat_list[1:], dim=0)
            # (1, N_views, 4, 4) --> (N_prev, N_views, 4, 4)
            ego2globals_curr = \
                ego2globals[0].repeat(self.num_frame - 1, 1, 1, 1)
            # (1, N_views, 4, 4) --> (N_prev, N_views, 4, 4)
            sensor2keyegos_curr = \
                sensor2keyegos[0].repeat(self.num_frame - 1, 1, 1, 1)
            ego2globals_prev = torch.cat(ego2globals[1:], dim=0)            # (N_prev, N_views, 4, 4)
            sensor2keyegos_prev = torch.cat(sensor2keyegos[1:], dim=0)      # (N_prev, N_views, 4, 4)
            bda_curr = bda.repeat(self.num_frame - 1, 1, 1)     # (N_prev, 3, 3)
            return feat_prev, [imgs[0],     # (1, N_views, 3, H, W)
                               sensor2keyegos_curr,     # (N_prev, N_views, 4, 4)
                               ego2globals_curr,        # (N_prev, N_views, 4, 4)
                               intrins[0],          # (1, N_views, 3, 3)
                               sensor2keyegos_prev,     # (N_prev, N_views, 4, 4)
                               ego2globals_prev,        # (N_prev, N_views, 4, 4)
                               post_rots[0],    # (1, N_views, 3, 3)
                               post_trans[0],   # (1, N_views, 3, )
                               bda_curr]        # (N_prev, 3, 3)

        if self.align_after_view_transfromation:
            for adj_id in range(1, self.num_frame):
                bev_feat_list[adj_id] = self.shift_feature(
                    bev_feat_list[adj_id],  # (B, C, Dy, Dx)
                    [sensor2keyegos[0],     # (B, N_views, 4, 4)
                     sensor2keyegos[adj_id]     # (B, N_views, 4, 4)
                    ],
                    bda     # (B, 3, 3)
                )   # (B, C, Dy, Dx)

        bev_feat = torch.cat(bev_feat_list, dim=1)      # (B, N_frames*C, Dy, Dx)
        x = self.bev_encoder(bev_feat)
        return [x], depth_list[0]


