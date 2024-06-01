_base_ = ['../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
          '../../../mmdetection3d/configs/_base_/default_runtime.py']

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
# point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
point_cloud_range = [-40.0, -40.0, -5.0, 40.0, 40.0, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 45.0, 0.5],
}

voxel_size = [0.1, 0.1, 0.2]
numC_Trans = 80
multi_adj_frame_id_cfg = (1, 1+1, 1)

model = dict(
    type='BEVDepth4DPano',
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch',
        pretrained='torchvision://resnet50',
    ),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=512,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVDepth',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=512,
        out_channels=numC_Trans,
        loss_depth_weight=1,
        depthnet_cfg=dict(use_dcn=False, aspp_mid_channels=96),
        downsample=16),
    pre_process=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_layer=[1, ],
        num_channels=[numC_Trans, ],
        stride=[1, ],
        backbone_output_ids=[0, ]),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg))+1),
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256),
    aux_centerness_head=dict(
        type='Centerness_Head',
        task_specific_weight=[1, 1, 0, 0, 0],
        in_channels=256,
        tasks=[
            dict(num_class=10, class_names=['car', 'truck',
                                            'construction_vehicle',
                                            'bus', 'trailer',
                                            'barrier',
                                            'motorcycle', 'bicycle',
                                            'pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.3, # 
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    occ_head=dict(
        type='BEVOCCHead2D_V2',
        in_dim=256,
        out_dim=256,
        Dz=16,
        use_mask=False,
        num_classes=18,
        use_predicter=True,
        class_balance=True,
        loss_occ=dict(
            type='CustomFocalLoss',
            use_sigmoid=True,
            loss_weight=1.0
        ),
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[800, 800, 40],
            voxel_size=voxel_size,
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])),
    test_cfg=dict(
        pts=dict(
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            pre_max_size=1000,
            post_max_size=500,

            # Scale-NMS
            nms_type=['rotate'],
            nms_thr=[0.2],
            nms_rescale_factor=[[1.0, 0.7, 0.7, 0.4, 0.55,
                                 1.1, 1.0, 1.0, 1.5, 3.5]]
        )
    ),
)

# Data
dataset_type = 'NuScenesDatasetOccpancy'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics',
                                'mask_lidar', 'mask_camera', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
        ])
]


input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    modality=input_modality,
    stereo=False,
    filter_empty_gt=False,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'bevdetv2-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[24, ])
runner = dict(type='EpochBasedRunner', max_epochs=24)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]

load_from = "ckpts/bevdet-r50-4d-depth-cbgs.pth"
# fp16 = dict(loss_scale='dynamic')
evaluation = dict(interval=1, start=20, pipeline=test_pipeline)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)



# ===> per class IoU of 6019 samples:
# ===> others - IoU = 10.89
# ===> barrier - IoU = 43.92
# ===> bicycle - IoU = 24.42
# ===> bus - IoU = 41.91
# ===> car - IoU = 45.18
# ===> construction_vehicle - IoU = 18.73
# ===> motorcycle - IoU = 25.59
# ===> pedestrian - IoU = 25.67
# ===> traffic_cone - IoU = 25.86
# ===> trailer - IoU = 25.29
# ===> truck - IoU = 31.84
# ===> driveable_surface - IoU = 59.03
# ===> other_flat - IoU = 31.53
# ===> sidewalk - IoU = 34.67
# ===> terrain - IoU = 31.49
# ===> manmade - IoU = 19.91
# ===> vegetation - IoU = 19.31
# ===> mIoU of 6019 samples: 30.31
# {'mIoU': array([0.109, 0.439, 0.244, 0.419, 0.452, 0.187, 0.256, 0.257, 0.259,
#        0.253, 0.318, 0.59 , 0.315, 0.347, 0.315, 0.199, 0.193, 0.835])}

# +----------------------+----------+----------+----------+
# |     Class Names      | RayIoU@1 | RayIoU@2 | RayIoU@4 |
# +----------------------+----------+----------+----------+
# |        others        |  0.094   |  0.105   |  0.107   |
# |       barrier        |  0.411   |  0.460   |  0.480   |
# |       bicycle        |  0.252   |  0.286   |  0.293   |
# |         bus          |  0.541   |  0.646   |  0.698   |
# |         car          |  0.520   |  0.594   |  0.621   |
# | construction_vehicle |  0.164   |  0.235   |  0.264   |
# |      motorcycle      |  0.212   |  0.305   |  0.321   |
# |      pedestrian      |  0.326   |  0.373   |  0.389   |
# |     traffic_cone     |  0.312   |  0.341   |  0.348   |
# |       trailer        |  0.220   |  0.291   |  0.372   |
# |        truck         |  0.430   |  0.520   |  0.565   |
# |  driveable_surface   |  0.552   |  0.633   |  0.720   |
# |      other_flat      |  0.293   |  0.330   |  0.361   |
# |       sidewalk       |  0.242   |  0.291   |  0.340   |
# |       terrain        |  0.236   |  0.305   |  0.369   |
# |       manmade        |  0.303   |  0.378   |  0.429   |
# |      vegetation      |  0.193   |  0.294   |  0.381   |
# +----------------------+----------+----------+----------+
# |         MEAN         |  0.312   |  0.376   |  0.415   |
# +----------------------+----------+----------+----------+
# 6019it [09:13, 10.87it/s]
# +----------------------+---------+---------+---------+
# |     Class Names      | RayPQ@1 | RayPQ@2 | RayPQ@4 |
# +----------------------+---------+---------+---------+
# |        others        |  0.020  |  0.028  |  0.030  |
# |       barrier        |  0.155  |  0.211  |  0.235  |
# |       bicycle        |  0.083  |  0.097  |  0.102  |
# |         bus          |  0.299  |  0.391  |  0.442  |
# |         car          |  0.277  |  0.360  |  0.384  |
# | construction_vehicle |  0.011  |  0.062  |  0.077  |
# |      motorcycle      |  0.098  |  0.149  |  0.166  |
# |      pedestrian      |  0.021  |  0.026  |  0.027  |
# |     traffic_cone     |  0.052  |  0.069  |  0.071  |
# |       trailer        |  0.043  |  0.062  |  0.071  |
# |        truck         |  0.158  |  0.248  |  0.293  |
# |  driveable_surface   |  0.440  |  0.559  |  0.680  |
# |      other_flat      |  0.065  |  0.089  |  0.107  |
# |       sidewalk       |  0.012  |  0.029  |  0.060  |
# |       terrain        |  0.009  |  0.028  |  0.053  |
# |       manmade        |  0.060  |  0.108  |  0.153  |
# |      vegetation      |  0.001  |  0.029  |  0.111  |
# +----------------------+---------+---------+---------+
# |         MEAN         |  0.106  |  0.150  |  0.180  |
# +----------------------+---------+---------+---------+
# {'RayIoU': 0.3676099569727112, 'RayIoU@1': 0.3118578145261225, 'RayIoU@2': 0.3757836068619914, 'RayIoU@4': 0.4151884495300196, 
#  'RayPQ': 0.14529917059571107, 'RayPQ@1': 0.1061843618020449, 'RayPQ@2': 0.14961373290314467, 'RayPQ@4': 0.18009941708194366}

