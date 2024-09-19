# Copyright (c) Phigent Robotics. All rights reserved.
from ...ops import TRTBEVPoolv2
from .bevdet import BEVDet
from .bevdepth import BEVDepth
from .bevdepth4d import BEVDepth4D
from .bevstereo4d import BEVStereo4D
from mmdet3d.models import DETECTORS
from mmdet3d.models.builder import build_head
import torch.nn.functional as F
from mmdet3d.core import bbox3d2result
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from ...ops import nearest_assign
# pool = ThreadPool(processes=4)  # 创建线程池

# for pano
grid_config_occ = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 45.0, 1.0],
}        
# det
det_class_name = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
    'barrier']

# occ
occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

det_ind = [2, 3, 4, 5, 6, 7, 9, 10]
occ_ind = [5, 3, 0, 4, 6, 7, 2, 1]
detind2occind = {
    0:4,
    1:10,
    2:9,
    3:3,
    4:5,
    5:2,
    6:6,
    7:7,
    8:8,
    9:1,
}
occind2detind = {
    4:0,
    10:1,
    9:2,
    3:3,
    5:4,
    2:5,
    6:6,
    7:7,
    8:8,
    1:9,
}
occind2detind_cuda = [-1, -1, 5, 3, 0, 4, 6, 7, -1, 2, 1]

inst_occ = np.ones([200, 200, 16])*0

import torch
X1, Y1, Z1 = 200, 200, 16
coords_x = torch.arange(X1).float()
coords_y = torch.arange(Y1).float()
coords_z = torch.arange(Z1).float()
coords = torch.stack(torch.meshgrid([coords_x, coords_y, coords_z])).permute(1, 2, 3, 0)  # W, H, D, 3
# coords = coords.cpu().numpy()
st = [grid_config_occ['x'][0], grid_config_occ['y'][0], grid_config_occ['z'][0]]
sx = [grid_config_occ['x'][2], grid_config_occ['y'][2], 0.4]

@DETECTORS.register_module()
class BEVDetOCC(BEVDet):
    def __init__(self,
                 occ_head=None,
                 upsample=False,
                 **kwargs):
        super(BEVDetOCC, self).__init__(**kwargs)
        self.occ_head = build_head(occ_head)
        self.pts_bbox_head = None
        self.upsample = upsample

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        losses = dict()
        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)

        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        loss_occ = self.forward_occ_train(occ_bev_feature, voxel_semantics, mask_camera)
        losses.update(loss_occ)
        return losses

    def forward_occ_train(self, img_feats, voxel_semantics, mask_camera):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
        """
        outs = self.occ_head(img_feats)
        # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.occ_head.loss(
            outs,  # (B, Dx, Dy, Dz, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
        )
        return loss_occ

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, _, _ = self.extract_feat(
            points, img_inputs=img, img_metas=img_metas, **kwargs)

        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        occ_list = self.simple_test_occ(occ_bev_feature, img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_list

    def simple_test_occ(self, img_feats, img_metas=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:

        Returns:
            occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        outs = self.occ_head(img_feats)
        if not hasattr(self.occ_head, "get_occ_gpu"):
            occ_preds = self.occ_head.get_occ(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        else:
            occ_preds = self.occ_head.get_occ_gpu(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_preds

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        outs = self.occ_head(occ_bev_feature)
        return outs


@DETECTORS.register_module()
class BEVDepthOCC(BEVDepth):
    def __init__(self,
                 occ_head=None,
                 upsample=False,
                 **kwargs):
        super(BEVDepthOCC, self).__init__(**kwargs)
        self.occ_head = build_head(occ_head)
        self.pts_bbox_head = None
        self.upsample = upsample

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        losses = dict()
        gt_depth = kwargs['gt_depth']   # (B, N_views, img_H, img_W)
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth

        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)

        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        loss_occ = self.forward_occ_train(occ_bev_feature, voxel_semantics, mask_camera)
        losses.update(loss_occ)
        return losses

    def forward_occ_train(self, img_feats, voxel_semantics, mask_camera):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
        """
        outs = self.occ_head(img_feats)
        # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.occ_head.loss(
            outs,  # (B, Dx, Dy, Dz, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
        )
        return loss_occ

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, _, _ = self.extract_feat(
            points, img_inputs=img, img_metas=img_metas, **kwargs)

        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        occ_list = self.simple_test_occ(occ_bev_feature, img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_list

    def simple_test_occ(self, img_feats, img_metas=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:

        Returns:
            occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        outs = self.occ_head(img_feats)
        # occ_preds = self.occ_head.get_occ(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        occ_preds = self.occ_head.get_occ_gpu(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_preds

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        outs = self.occ_head(occ_bev_feature)
        return outs


@DETECTORS.register_module()
class BEVDepthPano(BEVDepthOCC):
    def __init__(self,
                 aux_centerness_head=None,
                 **kwargs):
        super(BEVDepthPano, self).__init__(**kwargs)
        self.aux_centerness_head = None
        if aux_centerness_head:
            train_cfg = kwargs['train_cfg']
            test_cfg = kwargs['test_cfg']
            pts_train_cfg = train_cfg.pts if train_cfg else None
            aux_centerness_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            aux_centerness_head.update(test_cfg=pts_test_cfg)
            self.aux_centerness_head = build_head(aux_centerness_head)
        if 'inst_class_ids' in kwargs:
            self.inst_class_ids = kwargs['inst_class_ids']
        else:
            self.inst_class_ids = [2, 3, 4, 5, 6, 7, 9, 10]
            
        X1, Y1, Z1 = 200, 200, 16
        coords_x = torch.arange(X1).float()
        coords_y = torch.arange(Y1).float()
        coords_z = torch.arange(Z1).float()
        self.coords = torch.stack(torch.meshgrid([coords_x, coords_y, coords_z])).permute(1, 2, 3, 0)  # W, H, D, 3
        self.st = torch.tensor([grid_config_occ['x'][0], grid_config_occ['y'][0], grid_config_occ['z'][0]])
        self.sx = torch.tensor([grid_config_occ['x'][2], grid_config_occ['y'][2], 0.4])
        self.is_to_d = False

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        losses = dict()
        gt_depth = kwargs['gt_depth']   # (B, N_views, img_H, img_W)
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth

        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)

        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        loss_occ = self.forward_occ_train(occ_bev_feature, voxel_semantics, mask_camera)
        losses.update(loss_occ)
        
        losses_aux_centerness = self.forward_aux_centerness_train([occ_bev_feature], gt_bboxes_3d,
                                    gt_labels_3d, img_metas,
                                    gt_bboxes_ignore)
        losses.update(losses_aux_centerness)
        return losses
    
    def forward_aux_centerness_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        outs = self.aux_centerness_head(pts_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.aux_centerness_head.loss(*loss_inputs)
        return losses
        
    def simple_test_aux_centerness(self, x, img_metas, rescale=False, **kwargs):
        """Test function of point cloud branch."""
        # outs = self.aux_centerness_head(x)
        tx = self.aux_centerness_head.shared_conv(x[0])     # (B, C'=share_conv_channel, H, W)
        outs_inst_center_reg = self.aux_centerness_head.task_heads[0].reg(tx)
        outs_inst_center_height = self.aux_centerness_head.task_heads[0].height(tx)
        outs_inst_center_heatmap = self.aux_centerness_head.task_heads[0].heatmap(tx)
        outs = ([{
            "reg" : outs_inst_center_reg,
            "height" : outs_inst_center_height,
            "heatmap" : outs_inst_center_heatmap,
        }],)
                
        # # bbox_list = self.aux_centerness_head.get_bboxes(
        # #     outs, img_metas, rescale=rescale)
        # # bbox_results = [
        # #     bbox3d2result(bboxes, scores, labels)
        # #     for bboxes, scores, labels in bbox_list
        # # ]
        ins_cen_list = self.aux_centerness_head.get_centers(
            outs, img_metas, rescale=rescale)
        # return bbox_results, ins_cen_list
        return None, ins_cen_list

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        result_list = [dict() for _ in range(len(img_metas))]
        img_feats, _, _ = self.extract_feat(
            points, img_inputs=img, img_metas=img_metas, **kwargs)
        occ_bev_feature = img_feats[0]
        w_pano = kwargs['w_pano'] if 'w_pano' in kwargs else True
        if w_pano == True:
            bbox_pts, ins_cen_list = self.simple_test_aux_centerness([occ_bev_feature], img_metas, rescale=rescale, **kwargs)

        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        occ_list = self.simple_test_occ(occ_bev_feature, img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        for result_dict, occ_pred in zip(result_list, occ_list):
            result_dict['pred_occ'] = occ_pred
     
        w_panoproc = kwargs['w_panoproc'] if 'w_panoproc' in kwargs else True                # 37.53 ms
        if w_panoproc == True:
            # # for pano
            inst_xyz = ins_cen_list[0][0]
            if self.is_to_d == False:
                self.st = self.st.to(inst_xyz)
                self.sx = self.sx.to(inst_xyz)
                self.coords = self.coords.to(inst_xyz)
                self.is_to_d = True
            
            inst_xyz = ((inst_xyz - self.st) / self.sx).int()
            
            inst_cls = ins_cen_list[2][0].int()
            
            inst_num = 18                                                                    # 37.62 ms
            # inst_occ = torch.tensor(occ_pred).to(inst_cls)
            # inst_occ = occ_pred.clone().detach()
            inst_occ = occ_pred.clone().detach()                                             # 37.61 ms
            if len(inst_cls) > 0:
                cls_sort, indices = inst_cls.sort()
                l2s = {}
                if len(inst_cls) == 1:
                    l2s[cls_sort[0].item()] = 0
                l2s[cls_sort[0].item()] = 0
                # # tind_list = cls_sort[1:] - cls_sort[:-1]!=0
                # # for tind in range(len(tind_list)):
                # #     if tind_list[tind] == True:
                # #         l2s[cls_sort[1+tind].item()] = tind + 1
                tind_list = (cls_sort[1:] - cls_sort[:-1])!=0
                if tind_list.__len__() > 0:
                    for tind in torch.range(0,len(tind_list)-1)[tind_list]:
                        l2s[cls_sort[1+int(tind.item())].item()] = int(tind.item()) + 1

                is_cuda = True
                # is_cuda = False
                if is_cuda == True:
                    inst_id_list = indices + inst_num
                    l2s_key = indices.new_tensor([detind2occind[k] for k in l2s.keys()]).to(torch.int)
                    inst_occ = nearest_assign(
                        occ_pred.to(torch.int), 
                        l2s_key.to(torch.int),
                        indices.new_tensor(occind2detind_cuda).to(torch.int),
                        inst_cls.to(torch.int),
                        inst_xyz.to(torch.int),
                        inst_id_list.to(torch.int)
                        )
                else:
                    for cls_label_num_in_occ in self.inst_class_ids:
                        mask = occ_pred == cls_label_num_in_occ
                        if mask.sum() == 0:
                            continue
                        else:
                            cls_label_num_in_inst = occind2detind[cls_label_num_in_occ]
                            select_mask = inst_cls==cls_label_num_in_inst
                            if sum(select_mask) > 0:
                                indices = self.coords[mask]
                                inst_index_same_cls = inst_xyz[select_mask]
                                select_ind = ((indices[:,None,:] - inst_index_same_cls[None,:,:])**2).sum(-1).argmin(axis=1).int()
                                inst_occ[mask] = select_ind + inst_num + l2s[cls_label_num_in_inst]
                
            result_list[0]['pano_inst'] = inst_occ #.cpu().numpy()

        return result_list

@DETECTORS.register_module()
class BEVDepth4DOCC(BEVDepth4D):
    def __init__(self,
                 occ_head=None,
                 upsample=False,
                 **kwargs):
        super(BEVDepth4DOCC, self).__init__(**kwargs)
        self.occ_head = build_head(occ_head)
        self.pts_bbox_head = None
        self.upsample = upsample

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        gt_depth = kwargs['gt_depth']   # (B, N_views, img_H, img_W)
        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth

        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)
        loss_occ = self.forward_occ_train(img_feats[0], voxel_semantics, mask_camera)
        losses.update(loss_occ)
        return losses

    def forward_occ_train(self, img_feats, voxel_semantics, mask_camera):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
        """
        outs = self.occ_head(img_feats)
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.occ_head.loss(
            outs,  # (B, Dx, Dy, Dz, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
        )
        return loss_occ

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, _, _ = self.extract_feat(
            points, img_inputs=img, img_metas=img_metas, **kwargs)

        occ_list = self.simple_test_occ(img_feats[0], img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_list

    def simple_test_occ(self, img_feats, img_metas=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:

        Returns:
            occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        outs = self.occ_head(img_feats)
        # occ_preds = self.occ_head.get_occ(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        occ_preds = self.occ_head.get_occ_gpu(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_preds

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        outs = self.occ_head(occ_bev_feature)
        return outs

@DETECTORS.register_module()
class BEVDepth4DPano(BEVDepth4DOCC):
    def __init__(self,
                 aux_centerness_head=None,
                 **kwargs):
        super(BEVDepth4DPano, self).__init__(**kwargs)
        self.aux_centerness_head = None
        if aux_centerness_head:
            train_cfg = kwargs['train_cfg']
            test_cfg = kwargs['test_cfg']
            pts_train_cfg = train_cfg.pts if train_cfg else None
            aux_centerness_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            aux_centerness_head.update(test_cfg=pts_test_cfg)
            self.aux_centerness_head = build_head(aux_centerness_head)
        if 'inst_class_ids' in kwargs:
            self.inst_class_ids = kwargs['inst_class_ids']
        else:
            self.inst_class_ids = [2, 3, 4, 5, 6, 7, 9, 10]

        X1, Y1, Z1 = 200, 200, 16
        coords_x = torch.arange(X1).float()
        coords_y = torch.arange(Y1).float()
        coords_z = torch.arange(Z1).float()
        self.coords = torch.stack(torch.meshgrid([coords_x, coords_y, coords_z])).permute(1, 2, 3, 0)  # W, H, D, 3
        self.st = torch.tensor([grid_config_occ['x'][0], grid_config_occ['y'][0], grid_config_occ['z'][0]])
        self.sx = torch.tensor([grid_config_occ['x'][2], grid_config_occ['y'][2], 0.4])
        self.is_to_d = False

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        gt_depth = kwargs['gt_depth']   # (B, N_views, img_H, img_W)
        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth

        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)
        loss_occ = self.forward_occ_train(img_feats[0], voxel_semantics, mask_camera)
        losses.update(loss_occ)
        
        losses_aux_centerness = self.forward_aux_centerness_train([img_feats[0]], gt_bboxes_3d,
                                    gt_labels_3d, img_metas,
                                    gt_bboxes_ignore)
        losses.update(losses_aux_centerness)
        return losses
    
    def forward_aux_centerness_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        outs = self.aux_centerness_head(pts_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.aux_centerness_head.loss(*loss_inputs)
        return losses
        
    def simple_test_aux_centerness(self, x, img_metas, rescale=False, **kwargs):
        """Test function of point cloud branch."""
        outs = self.aux_centerness_head(x)
        bbox_list = self.aux_centerness_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        ins_cen_list = self.aux_centerness_head.get_centers(
            outs, img_metas, rescale=rescale)
        return bbox_results, ins_cen_list

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        result_list = [dict() for _ in range(len(img_metas))]
        img_feats, _, _ = self.extract_feat(
            points, img_inputs=img, img_metas=img_metas, **kwargs)
        occ_bev_feature = img_feats[0]
        w_pano = kwargs['w_pano'] if 'w_pano' in kwargs else True
        if w_pano == True:
            bbox_pts, ins_cen_list = self.simple_test_aux_centerness([occ_bev_feature], img_metas, rescale=rescale, **kwargs)

        occ_list = self.simple_test_occ(occ_bev_feature, img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        for result_dict, occ_pred in zip(result_list, occ_list):
            result_dict['pred_occ'] = occ_pred

        w_panoproc = kwargs['w_panoproc'] if 'w_panoproc' in kwargs else True
        if w_panoproc == True:
            # # for pano
            inst_xyz = ins_cen_list[0][0]
            if self.is_to_d == False:
                self.st = self.st.to(inst_xyz)
                self.sx = self.sx.to(inst_xyz)
                self.coords = self.coords.to(inst_xyz)
                self.is_to_d = True
            
            inst_xyz = ((inst_xyz - self.st) / self.sx).int()
            
            inst_cls = ins_cen_list[2][0].int()
            
            inst_num = 18                                                                    # 37.62 ms
            # inst_occ = torch.tensor(occ_pred).to(inst_cls)
            # inst_occ = occ_pred.clone().detach()
            inst_occ = occ_pred.clone().detach()                                             # 37.61 ms
            if len(inst_cls) > 0:
                cls_sort, indices = inst_cls.sort()
                l2s = {}
                if len(inst_cls) == 1:
                    l2s[cls_sort[0].item()] = 0
                l2s[cls_sort[0].item()] = 0
                # # tind_list = cls_sort[1:] - cls_sort[:-1]!=0
                # # for tind in range(len(tind_list)):
                # #     if tind_list[tind] == True:
                # #         l2s[cls_sort[1+tind].item()] = tind + 1
                tind_list = (cls_sort[1:] - cls_sort[:-1])!=0
                if tind_list.__len__() > 0:
                    for tind in torch.range(0,len(tind_list)-1)[tind_list]:
                        l2s[cls_sort[1+int(tind.item())].item()] = int(tind.item()) + 1

                is_cuda = True
                # is_cuda = False
                if is_cuda == True:
                    inst_id_list = indices + inst_num
                    l2s_key = indices.new_tensor([detind2occind[k] for k in l2s.keys()]).to(torch.int)
                    inst_occ = nearest_assign(
                        occ_pred.to(torch.int), 
                        l2s_key.to(torch.int),
                        indices.new_tensor(occind2detind_cuda).to(torch.int),
                        inst_cls.to(torch.int),
                        inst_xyz.to(torch.int),
                        inst_id_list.to(torch.int)
                        )
                else:
                    for cls_label_num_in_occ in self.inst_class_ids:
                        mask = occ_pred == cls_label_num_in_occ
                        if mask.sum() == 0:
                            continue
                        else:
                            cls_label_num_in_inst = occind2detind[cls_label_num_in_occ]
                            select_mask = inst_cls==cls_label_num_in_inst
                            if sum(select_mask) > 0:
                                indices = self.coords[mask]
                                inst_index_same_cls = inst_xyz[select_mask]
                                select_ind = ((indices[:,None,:] - inst_index_same_cls[None,:,:])**2).sum(-1).argmin(axis=1).int()
                                inst_occ[mask] = select_ind + inst_num + l2s[cls_label_num_in_inst]
                
            result_list[0]['pano_inst'] = inst_occ #.cpu().numpy()

        return result_list

@DETECTORS.register_module()
class BEVStereo4DOCC(BEVStereo4D):
    def __init__(self,
                 occ_head=None,
                 upsample=False,
                 **kwargs):
        super(BEVStereo4DOCC, self).__init__(**kwargs)
        self.occ_head = build_head(occ_head)
        self.pts_bbox_head = None
        self.upsample = upsample

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        gt_depth = kwargs['gt_depth']   # (B, N_views, img_H, img_W)
        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth

        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)
        loss_occ = self.forward_occ_train(img_feats[0], voxel_semantics, mask_camera)
        losses.update(loss_occ)
        return losses

    def forward_occ_train(self, img_feats, voxel_semantics, mask_camera):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
        """
        outs = self.occ_head(img_feats)
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.occ_head.loss(
            outs,  # (B, Dx, Dy, Dz, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
        )
        return loss_occ

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, _, _ = self.extract_feat(
            points, img_inputs=img, img_metas=img_metas, **kwargs)

        occ_list = self.simple_test_occ(img_feats[0], img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_list

    def simple_test_occ(self, img_feats, img_metas=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:

        Returns:
            occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        outs = self.occ_head(img_feats)
        # occ_preds = self.occ_head.get_occ(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        occ_preds = self.occ_head.get_occ_gpu(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_preds

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        outs = self.occ_head(occ_bev_feature)
        return outs
    

@DETECTORS.register_module()
class BEVDetOCCTRT(BEVDetOCC):
    def __init__(self,
                 wocc=True,
                 wdet3d=True,
                 uni_train=True,
                 **kwargs):
        super(BEVDetOCCTRT, self).__init__(**kwargs)
        self.wocc = wocc
        self.wdet3d = wdet3d
        self.uni_train = uni_train
        
    def result_serialize(self, outs_det3d=None, outs_occ=None):
        outs_ = []
        if outs_det3d is not None:
            for out in outs_det3d:
                for key in ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']:
                    outs_.append(out[0][key])
        if outs_occ is not None:
            outs_.append(outs_occ)
        return outs_

    def result_deserialize(self, outs):
        outs_ = []
        keys = ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']
        for head_id in range(len(outs) // 6):
            outs_head = [dict()]
            for kid, key in enumerate(keys):
                outs_head[0][key] = outs[head_id * 6 + kid]
            outs_.append(outs_head)
        return outs_

    def forward_part1(
        self,
        img,
    ):
        x = self.img_backbone(img)
        x = self.img_neck(x)
        x = self.img_view_transformer.depth_net(x[0])
        depth = x[:, :self.img_view_transformer.D].softmax(dim=1)
        tran_feat = x[:, self.img_view_transformer.D:(
            self.img_view_transformer.D +
            self.img_view_transformer.out_channels)]
        tran_feat = tran_feat.permute(0, 2, 3, 1)
        # depth = depth.reshape(-1)
        # tran_feat = tran_feat.flatten(0,2)
        return tran_feat.flatten(0,2), depth.reshape(-1)

    def forward_part2(
        self,
        tran_feat,
        depth,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        interval_starts,
        interval_lengths,
    ):
        tran_feat = tran_feat.reshape(6, 16, 44, 64)
        depth = depth.reshape(6, 16, 44, 44)
        x = TRTBEVPoolv2.apply(depth.contiguous(), tran_feat.contiguous(),
                               ranks_depth, ranks_feat, ranks_bev,
                               interval_starts, interval_lengths,
                               int(self.img_view_transformer.grid_size[0].item()),
                               int(self.img_view_transformer.grid_size[1].item()),
                               int(self.img_view_transformer.grid_size[2].item())
                               ) # -> [1, 64, 200, 200]
        return x.reshape(-1)

    def forward_part3(
        self,
        x
    ):
        x = x.reshape(1, 200, 200, 64)
        x = x.permute(0, 3, 1, 2).contiguous()
        # return [x, 2*x, 3*x, 4*x, 5*x, 6*x, 7*x]
        bev_feature = self.img_bev_encoder_backbone(x)
        occ_bev_feature = self.img_bev_encoder_neck(bev_feature)

        outs_occ = None
        if self.wocc == True:
            if self.uni_train == True:
                if self.upsample:
                    occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                                    mode='bilinear', align_corners=True)
            outs_occ = self.occ_head(occ_bev_feature)

        outs_det3d = None
        if self.wdet3d == True:
            outs_det3d = self.pts_bbox_head([occ_bev_feature])

        outs = self.result_serialize(outs_det3d, outs_occ)
        return outs
    
    def forward_ori(
        self,
        img,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        interval_starts,
        interval_lengths,
    ):
        x = self.img_backbone(img)
        x = self.img_neck(x)
        x = self.img_view_transformer.depth_net(x[0])
        depth = x[:, :self.img_view_transformer.D].softmax(dim=1)
        tran_feat = x[:, self.img_view_transformer.D:(
            self.img_view_transformer.D +
            self.img_view_transformer.out_channels)]
        tran_feat = tran_feat.permute(0, 2, 3, 1)
        x = TRTBEVPoolv2.apply(depth.contiguous(), tran_feat.contiguous(),
                               ranks_depth, ranks_feat, ranks_bev,
                               interval_starts, interval_lengths,
                               int(self.img_view_transformer.grid_size[0].item()),
                               int(self.img_view_transformer.grid_size[1].item()),
                               int(self.img_view_transformer.grid_size[2].item())
                               )
        x = x.permute(0, 3, 1, 2).contiguous()
        # return [x, 2*x, 3*x, 4*x, 5*x, 6*x, 7*x]
        bev_feature = self.img_bev_encoder_backbone(x)
        occ_bev_feature = self.img_bev_encoder_neck(bev_feature)

        outs_occ = None
        if self.wocc == True:
            if self.uni_train == True:
                if self.upsample:
                    occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                                    mode='bilinear', align_corners=True)
            outs_occ = self.occ_head(occ_bev_feature)

        outs_det3d = None
        if self.wdet3d == True:
            outs_det3d = self.pts_bbox_head([occ_bev_feature])

        outs = self.result_serialize(outs_det3d, outs_occ)
        return outs

    def forward_with_argmax(
        self,
        img,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        interval_starts,
        interval_lengths,
    ):

        outs = self.forward_ori(
                img,
                ranks_depth,
                ranks_feat,
                ranks_bev,
                interval_starts,
                interval_lengths,
            )
        pred_occ_label = outs[0].argmax(-1)
        return pred_occ_label


    def get_bev_pool_input(self, input):
        input = self.prepare_inputs(input)
        coor = self.img_view_transformer.get_lidar_coor(*input[1:7])
        return self.img_view_transformer.voxel_pooling_prepare_v2(coor)


@DETECTORS.register_module()
class BEVDepthOCCTRT(BEVDetOCC):
    def __init__(self,
                 wocc=True,
                 wdet3d=True,
                 uni_train=True,
                 **kwargs):
        super(BEVDepthOCCTRT, self).__init__(**kwargs)
        self.wocc = wocc
        self.wdet3d = wdet3d
        self.uni_train = uni_train
        
    def result_serialize(self, outs_det3d=None, outs_occ=None):
        outs_ = []
        if outs_det3d is not None:
            for out in outs_det3d:
                for key in ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']:
                    outs_.append(out[0][key])
        if outs_occ is not None:
            outs_.append(outs_occ)
        return outs_

    def result_deserialize(self, outs):
        outs_ = []
        keys = ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']
        for head_id in range(len(outs) // 6):
            outs_head = [dict()]
            for kid, key in enumerate(keys):
                outs_head[0][key] = outs[head_id * 6 + kid]
            outs_.append(outs_head)
        return outs_

    def forward_ori(
        self,
        img,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        interval_starts,
        interval_lengths,
        mlp_input,
    ):
        x = self.img_backbone(img)
        x = self.img_neck(x)
        x = self.img_view_transformer.depth_net(x[0], mlp_input)
        depth = x[:, :self.img_view_transformer.D].softmax(dim=1)
        tran_feat = x[:, self.img_view_transformer.D:(
            self.img_view_transformer.D +
            self.img_view_transformer.out_channels)]
        tran_feat = tran_feat.permute(0, 2, 3, 1)
        x = TRTBEVPoolv2.apply(depth.contiguous(), tran_feat.contiguous(),
                               ranks_depth, ranks_feat, ranks_bev,
                               interval_starts, interval_lengths,
                               int(self.img_view_transformer.grid_size[0].item()),
                               int(self.img_view_transformer.grid_size[1].item()),
                               int(self.img_view_transformer.grid_size[2].item())
                               )
        x = x.permute(0, 3, 1, 2).contiguous()
        # return [x, 2*x, 3*x, 4*x, 5*x, 6*x, 7*x]
        bev_feature = self.img_bev_encoder_backbone(x)
        occ_bev_feature = self.img_bev_encoder_neck(bev_feature)

        outs_occ = None
        if self.wocc == True:
            if self.uni_train == True:
                if self.upsample:
                    occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                                    mode='bilinear', align_corners=True)
            outs_occ = self.occ_head(occ_bev_feature)

        outs_det3d = None
        if self.wdet3d == True:
            outs_det3d = self.pts_bbox_head([occ_bev_feature])

        outs = self.result_serialize(outs_det3d, outs_occ)
        return outs

    def forward_with_argmax(
        self,
        img,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        interval_starts,
        interval_lengths,
        mlp_input,
    ):

        outs = self.forward_ori(
                img,
                ranks_depth,
                ranks_feat,
                ranks_bev,
                interval_starts,
                interval_lengths,
                mlp_input,
            )
        pred_occ_label = outs[0].argmax(-1)
        return pred_occ_label


    def get_bev_pool_input(self, input):
        input = self.prepare_inputs(input)
        coor = self.img_view_transformer.get_lidar_coor(*input[1:7])
        mlp_input = self.img_view_transformer.get_mlp_input(*input[1:7])
                # sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda)  # (B, N_views, 27)
        return self.img_view_transformer.voxel_pooling_prepare_v2(coor), mlp_input


@DETECTORS.register_module()
class BEVDepthPanoTRT(BEVDepthPano):
    def __init__(self,
                 wocc=True,
                 wdet3d=True,
                 uni_train=True,
                 **kwargs):
        super(BEVDepthPanoTRT, self).__init__(**kwargs)
        self.wocc = wocc
        self.wdet3d = wdet3d
        self.uni_train = uni_train
        
    def result_serialize(self, outs_det3d=None, outs_occ=None):
        outs_ = []
        if outs_det3d is not None:
            for out in outs_det3d:
                for key in ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']:
                    outs_.append(out[0][key])
        if outs_occ is not None:
            outs_.append(outs_occ)
        return outs_

    def result_deserialize(self, outs):
        outs_ = []
        keys = ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']
        for head_id in range(len(outs) // 6):
            outs_head = [dict()]
            for kid, key in enumerate(keys):
                outs_head[0][key] = outs[head_id * 6 + kid]
            outs_.append(outs_head)
        return outs_

    def forward_part1(
        self,
        img,
        mlp_input,
    ):
        x = self.img_backbone(img)
        x = self.img_neck(x)
        x = self.img_view_transformer.depth_net(x[0], mlp_input)
        depth = x[:, :self.img_view_transformer.D].softmax(dim=1)
        tran_feat = x[:, self.img_view_transformer.D:(
            self.img_view_transformer.D +
            self.img_view_transformer.out_channels)]
        tran_feat = tran_feat.permute(0, 2, 3, 1)
        # depth = depth.reshape(-1)
        # tran_feat = tran_feat.flatten(0,2)
        return tran_feat.flatten(0,2), depth.reshape(-1)

    def forward_part2(
        self,
        tran_feat,
        depth,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        interval_starts,
        interval_lengths,
    ):
        tran_feat = tran_feat.reshape(6, 16, 44, 64)
        depth = depth.reshape(6, 16, 44, 44)
        x = TRTBEVPoolv2.apply(depth.contiguous(), tran_feat.contiguous(),
                               ranks_depth, ranks_feat, ranks_bev,
                               interval_starts, interval_lengths,
                               int(self.img_view_transformer.grid_size[0].item()),
                               int(self.img_view_transformer.grid_size[1].item()),
                               int(self.img_view_transformer.grid_size[2].item())
                               ) # -> [1, 64, 200, 200]
        return x.reshape(-1)

    def forward_part3(
        self,
        x
    ):
        x = x.reshape(1, 200, 200, 64)
        x = x.permute(0, 3, 1, 2).contiguous()
        # return [x, 2*x, 3*x, 4*x, 5*x, 6*x, 7*x]
        bev_feature = self.img_bev_encoder_backbone(x)
        occ_bev_feature = self.img_bev_encoder_neck(bev_feature)

        outs_occ = None
        if self.wocc == True:
            if self.uni_train == True:
                if self.upsample:
                    occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                                    mode='bilinear', align_corners=True)
            outs_occ = self.occ_head(occ_bev_feature)

        outs_det3d = None
        if self.wdet3d == True:
            outs_det3d = self.pts_bbox_head([occ_bev_feature])
        outs = self.result_serialize(outs_det3d, outs_occ)
        
        # outs_inst_center = self.aux_centerness_head([occ_bev_feature])
        x = self.aux_centerness_head.shared_conv(occ_bev_feature)     # (B, C'=share_conv_channel, H, W)
        # 运行不同task_head,
        outs_inst_center_reg = self.aux_centerness_head.task_heads[0].reg(x)
        outs.append(outs_inst_center_reg)
        outs_inst_center_height = self.aux_centerness_head.task_heads[0].height(x)
        outs.append(outs_inst_center_height)
        outs_inst_center_heatmap = self.aux_centerness_head.task_heads[0].heatmap(x)
        outs.append(outs_inst_center_heatmap)

    def forward_ori(
        self,
        img,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        interval_starts,
        interval_lengths,
        mlp_input,
    ):
        x = self.img_backbone(img)
        x = self.img_neck(x)
        x = self.img_view_transformer.depth_net(x[0], mlp_input)
        depth = x[:, :self.img_view_transformer.D].softmax(dim=1)
        tran_feat = x[:, self.img_view_transformer.D:(
            self.img_view_transformer.D +
            self.img_view_transformer.out_channels)]
        tran_feat = tran_feat.permute(0, 2, 3, 1)
        x = TRTBEVPoolv2.apply(depth.contiguous(), tran_feat.contiguous(),
                               ranks_depth, ranks_feat, ranks_bev,
                               interval_starts, interval_lengths,
                               int(self.img_view_transformer.grid_size[0].item()),
                               int(self.img_view_transformer.grid_size[1].item()),
                               int(self.img_view_transformer.grid_size[2].item())
                               )
        x = x.permute(0, 3, 1, 2).contiguous()
        # return [x, 2*x, 3*x, 4*x, 5*x, 6*x, 7*x]
        bev_feature = self.img_bev_encoder_backbone(x)
        occ_bev_feature = self.img_bev_encoder_neck(bev_feature)

        outs_occ = None
        if self.wocc == True:
            if self.uni_train == True:
                if self.upsample:
                    occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                                    mode='bilinear', align_corners=True)
            outs_occ = self.occ_head(occ_bev_feature)

        outs_det3d = None
        if self.wdet3d == True:
            outs_det3d = self.pts_bbox_head([occ_bev_feature])
        outs = self.result_serialize(outs_det3d, outs_occ)

        # outs_inst_center = self.aux_centerness_head([occ_bev_feature])
        x = self.aux_centerness_head.shared_conv(occ_bev_feature)     # (B, C'=share_conv_channel, H, W)
        # 运行不同task_head,
        outs_inst_center_reg = self.aux_centerness_head.task_heads[0].reg(x)
        outs.append(outs_inst_center_reg)
        outs_inst_center_height = self.aux_centerness_head.task_heads[0].height(x)
        outs.append(outs_inst_center_height)
        outs_inst_center_heatmap = self.aux_centerness_head.task_heads[0].heatmap(x)
        outs.append(outs_inst_center_heatmap)

        return outs

    def forward_with_argmax(
        self,
        img,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        interval_starts,
        interval_lengths,
        mlp_input,
    ):

        outs = self.forward_ori(
                img,
                ranks_depth,
                ranks_feat,
                ranks_bev,
                interval_starts,
                interval_lengths,
                mlp_input,
            )
        pred_occ_label = outs[0].argmax(-1)
        return pred_occ_label, *outs[1:]

    def get_bev_pool_input(self, input):
        input = self.prepare_inputs(input)
        coor = self.img_view_transformer.get_lidar_coor(*input[1:7])
        mlp_input = self.img_view_transformer.get_mlp_input(*input[1:7])
                # sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda)  # (B, N_views, 27)
        return self.img_view_transformer.voxel_pooling_prepare_v2(coor), mlp_input
