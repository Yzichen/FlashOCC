# Copyright (c) OpenMMLab. All rights reserved.
import os
import mmcv
import torch
import cv2
import numpy as np
from tqdm import tqdm

from mmdet3d.datasets import DATASETS
from .nuscenes_dataset_bevdet import NuScenesDatasetBEVDet as NuScenesDataset
from ..core.evaluation.occ_metrics import Metric_mIoU, Metric_FScore
from .ego_pose_dataset import EgoPoseDataset
from ..core.evaluation.ray_metrics import main as calc_rayiou
from torch.utils.data import DataLoader
from ..core.evaluation.ray_metrics import main_raypq
import torch
import glob


colors_map = np.array(
    [
        [0,   0,   0, 255],  # 0 undefined
        [255, 158, 0, 255],  # 1 car  orange
        [0, 0, 230, 255],    # 2 pedestrian  Blue
        [47, 79, 79, 255],   # 3 sign  Darkslategrey
        [220, 20, 60, 255],  # 4 CYCLIST  Crimson
        [255, 69, 0, 255],   # 5 traiffic_light  Orangered
        [255, 140, 0, 255],  # 6 pole  Darkorange
        [233, 150, 70, 255], # 7 construction_cone  Darksalmon
        [255, 61, 99, 255],  # 8 bycycle  Red
        [112, 128, 144, 255],# 9 motorcycle  Slategrey
        [222, 184, 135, 255],# 10 building Burlywood
        [0, 175, 0, 255],    # 11 vegetation  Green
        [165, 42, 42, 255],  # 12 trunk  nuTonomy green
        [0, 207, 191, 255],  # 13 curb, road, lane_marker, other_ground
        [75, 0, 75, 255], # 14 walkable, sidewalk
        [255, 0, 0, 255], # 15 unobsrvd
        [0, 0, 0, 0],  # 16 undefined
        [0, 0, 0, 0],  # 16 undefined
    ])


@DATASETS.register_module()
class NuScenesDatasetOccpancy(NuScenesDataset):
    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        input_dict = super(NuScenesDatasetOccpancy, self).get_data_info(index)
        # standard protocol modified from SECOND.Pytorch
        # input_dict['occ_gt_path'] = os.path.join(self.data_root, self.data_infos[index]['occ_path'])
        input_dict['occ_gt_path'] = self.data_infos[index]['occ_path']
        return input_dict

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        metric = eval_kwargs['metric'][0]
        print("metric = ", metric)
        if metric == 'ray-iou':
            occ_gts = []
            occ_preds = []
            lidar_origins = []
            inst_gts = []
            inst_preds = []
            
            print('\nStarting Evaluation...')

            data_loader = DataLoader(
                EgoPoseDataset(self.data_infos),
                batch_size=1,
                shuffle=False,
                num_workers=8
            )

            sample_tokens = [info['token'] for info in self.data_infos]

            for i, batch in enumerate(data_loader):
                # if i > 5:
                #     break
                token = batch[0][0]
                output_origin = batch[1]

                data_id = sample_tokens.index(token)
                info = self.data_infos[data_id]
                # occ_gt = np.load(os.path.join(self.data_root, info['occ_path'], 'labels.npz'))
                # occ_gt = np.load(os.path.join(info['occ_path'], 'labels.npz'))
                occ_gt = np.load(os.path.join(info['occ_path'].replace('data/nuscenes/gts/', 'data/nuscenes/occ3d_panoptic/'), 'labels.npz'))
                gt_semantics = occ_gt['semantics']      # (Dx, Dy, Dz)
                mask_lidar = occ_gt['mask_lidar'].astype(bool)      # (Dx, Dy, Dz)
                mask_camera = occ_gt['mask_camera'].astype(bool)    # (Dx, Dy, Dz)
                occ_pred = occ_results[data_id]['pred_occ'].cpu().numpy()     # (Dx, Dy, Dz)
                # occ_pred = occ_results[data_id]['pred_occ']     # (Dx, Dy, Dz)

                lidar_origins.append(output_origin)
                occ_gts.append(gt_semantics)
                occ_preds.append(occ_pred)

                if 'pano_inst' in occ_results[data_id].keys():
                    pano_inst = occ_results[data_id]['pano_inst'].cpu()
                    # pano_inst = torch.from_numpy(occ_results[data_id]['pano_inst'])
                    pano_inst = pano_inst.squeeze(0).numpy()
                    gt_instances = occ_gt['instances']
                    inst_gts.append(gt_instances)
                    inst_preds.append(pano_inst)
                    
            eval_results = calc_rayiou(occ_preds, occ_gts, lidar_origins)
            if len(inst_preds) > 0:
                eval_results.update(main_raypq(occ_preds, occ_gts, inst_preds, inst_gts, lidar_origins))
            # eval_results = main_raypq(occ_preds, occ_gts, inst_preds, inst_gts, lidar_origins)
        else:
            self.occ_eval_metrics = Metric_mIoU(
                num_classes=18,
                use_lidar_mask=False,
                use_image_mask=True)

            print('\nStarting Evaluation...')
            for index, occ_pred in enumerate(tqdm(occ_results)):
                # occ_pred: (Dx, Dy, Dz)
                info = self.data_infos[index]
                # occ_gt = np.load(os.path.join(self.data_root, info['occ_path'], 'labels.npz'))
                occ_gt = np.load(os.path.join(info['occ_path'], 'labels.npz'))
                gt_semantics = occ_gt['semantics']      # (Dx, Dy, Dz)
                mask_lidar = occ_gt['mask_lidar'].astype(bool)      # (Dx, Dy, Dz)
                mask_camera = occ_gt['mask_camera'].astype(bool)    # (Dx, Dy, Dz)
                # occ_pred = occ_pred
                self.occ_eval_metrics.add_batch(
                    occ_pred['pred_occ'] if (isinstance(occ_pred, dict) and 'pred_occ' in occ_pred) else occ_pred,   # (Dx, Dy, Dz)
                    gt_semantics,   # (Dx, Dy, Dz)
                    mask_lidar,     # (Dx, Dy, Dz)
                    mask_camera     # (Dx, Dy, Dz)
                )

                # if index % 100 == 0 and show_dir is not None:
                #     gt_vis = self.vis_occ(gt_semantics)
                #     pred_vis = self.vis_occ(occ_pred)
                #     mmcv.imwrite(np.concatenate([gt_vis, pred_vis], axis=1),
                #                  os.path.join(show_dir + "%d.jpg"%index))
                
                if show_dir is not None:
                    mmcv.mkdir_or_exist(show_dir)
                    # scene_name = info['scene_name']
                    scene_name = [tem for tem in info['occ_path'].split('/') if 'scene-' in tem][0]
                    sample_token = info['token']
                    mmcv.mkdir_or_exist(os.path.join(show_dir, scene_name, sample_token))
                    save_path = os.path.join(show_dir, scene_name, sample_token, 'pred.npz')
                    np.savez_compressed(save_path, pred=occ_pred['pred_occ'] if (isinstance(occ_pred, dict) and 'pred_occ' in occ_pred) else occ_pred, gt=occ_gt, sample_token=sample_token)

            eval_results = self.occ_eval_metrics.count_miou()

        return eval_results


    def vis_occ(self, semantics):
        # simple visualization of result in BEV
        semantics_valid = np.logical_not(semantics == 17)
        d = np.arange(16).reshape(1, 1, 16)
        d = np.repeat(d, 200, axis=0)
        d = np.repeat(d, 200, axis=1).astype(np.float32)
        d = d * semantics_valid
        selected = np.argmax(d, axis=2)

        selected_torch = torch.from_numpy(selected)
        semantics_torch = torch.from_numpy(semantics)

        occ_bev_torch = torch.gather(semantics_torch, dim=2,
                                     index=selected_torch.unsqueeze(-1))
        occ_bev = occ_bev_torch.numpy()

        occ_bev = occ_bev.flatten().astype(np.int32)
        occ_bev_vis = colors_map[occ_bev].astype(np.uint8)
        occ_bev_vis = occ_bev_vis.reshape(200, 200, 4)[::-1, ::-1, :3]
        occ_bev_vis = cv2.resize(occ_bev_vis,(400,400))
        return occ_bev_vis
