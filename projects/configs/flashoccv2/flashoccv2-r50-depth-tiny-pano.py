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
    'depth': [1.0, 45.0, 1.0],
}

voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 64

model = dict(
    type='BEVDepthPano',     # single-frame
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
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVDepth',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        loss_depth_weight=1,
        depthnet_cfg=dict(use_dcn=False, aspp_mid_channels=96),
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=128),
    aux_centerness_head=dict(
        type='Centerness_Head',
        task_specific_weight=[1, 1, 0, 0, 0],
        in_channels=128,
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
        in_dim=128,
        out_dim=128,
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
        sequential=False),
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
    dict(type='PrepareImageInputs', data_config=data_config, sequential=False),
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
    img_info_prototype='bevdet',
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
# ===> others - IoU = 10.33
# ===> barrier - IoU = 41.02
# ===> bicycle - IoU = 22.16
# ===> bus - IoU = 39.75
# ===> car - IoU = 42.63
# ===> construction_vehicle - IoU = 20.53
# ===> motorcycle - IoU = 24.01
# ===> pedestrian - IoU = 23.71
# ===> traffic_cone - IoU = 24.65
# ===> trailer - IoU = 25.58
# ===> truck - IoU = 30.63
# ===> driveable_surface - IoU = 58.0
# ===> other_flat - IoU = 32.12
# ===> sidewalk - IoU = 33.78
# ===> terrain - IoU = 31.02
# ===> manmade - IoU = 17.67
# ===> vegetation - IoU = 17.74
# ===> mIoU of 6019 samples: 29.14
# {'mIoU': array([0.103, 0.41 , 0.222, 0.397, 0.426, 0.205, 0.24 , 0.237, 0.246,
#        0.256, 0.306, 0.58 , 0.321, 0.338, 0.31 , 0.177, 0.177, 0.832])}

# +----------------------+----------+----------+----------+
# |     Class Names      | RayIoU@1 | RayIoU@2 | RayIoU@4 |
# +----------------------+----------+----------+----------+
# |        others        |  0.095   |  0.107   |  0.110   |
# |       barrier        |  0.374   |  0.429   |  0.452   |
# |       bicycle        |  0.208   |  0.242   |  0.248   |
# |         bus          |  0.498   |  0.603   |  0.659   |
# |         car          |  0.489   |  0.568   |  0.598   |
# | construction_vehicle |  0.171   |  0.247   |  0.279   |
# |      motorcycle      |  0.190   |  0.277   |  0.298   |
# |      pedestrian      |  0.295   |  0.344   |  0.361   |
# |     traffic_cone     |  0.290   |  0.324   |  0.332   |
# |       trailer        |  0.207   |  0.292   |  0.368   |
# |        truck         |  0.411   |  0.507   |  0.551   |
# |  driveable_surface   |  0.531   |  0.614   |  0.704   |
# |      other_flat      |  0.286   |  0.325   |  0.357   |
# |       sidewalk       |  0.234   |  0.280   |  0.328   |
# |       terrain        |  0.220   |  0.290   |  0.356   |
# |       manmade        |  0.267   |  0.343   |  0.392   |
# |      vegetation      |  0.174   |  0.272   |  0.358   |
# +----------------------+----------+----------+----------+
# |         MEAN         |  0.291   |  0.357   |  0.397   |
# +----------------------+----------+----------+----------+
# 6019it [09:34, 10.48it/s]
# +----------------------+---------+---------+---------+
# |     Class Names      | RayPQ@1 | RayPQ@2 | RayPQ@4 |
# +----------------------+---------+---------+---------+
# |        others        |  0.017  |  0.024  |  0.025  |
# |       barrier        |  0.107  |  0.169  |  0.204  |
# |       bicycle        |  0.069  |  0.086  |  0.088  |
# |         bus          |  0.244  |  0.350  |  0.408  |
# |         car          |  0.238  |  0.326  |  0.352  |
# | construction_vehicle |  0.018  |  0.081  |  0.105  |
# |      motorcycle      |  0.061  |  0.105  |  0.117  |
# |      pedestrian      |  0.016  |  0.022  |  0.023  |
# |     traffic_cone     |  0.030  |  0.049  |  0.052  |
# |       trailer        |  0.029  |  0.047  |  0.056  |
# |        truck         |  0.151  |  0.240  |  0.286  |
# |  driveable_surface   |  0.407  |  0.531  |  0.662  |
# |      other_flat      |  0.054  |  0.078  |  0.098  |
# |       sidewalk       |  0.009  |  0.030  |  0.061  |
# |       terrain        |  0.006  |  0.022  |  0.045  |
# |       manmade        |  0.044  |  0.091  |  0.128  |
# |      vegetation      |  0.001  |  0.021  |  0.091  |
# +----------------------+---------+---------+---------+
# |         MEAN         |  0.088  |  0.134  |  0.165  |
# +----------------------+---------+---------+---------+
# {'RayIoU': 0.34819957391233375, 'RayIoU@1': 0.29065973127346445, 'RayIoU@2': 0.3566749015912661, 'RayIoU@4': 0.39726408887227066, 
#  'RayPQ': 0.12890890185841564, 'RayPQ@1': 0.08832135839934552, 'RayPQ@2': 0.1336058084882046, 'RayPQ@4': 0.1647995386876968}
