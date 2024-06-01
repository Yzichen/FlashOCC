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
multi_adj_frame_id_cfg = (1, 8+1, 1)

model = dict(
    type='BEVDepth4DPano',
    num_adj=multi_adj_frame_id_cfg[1]-1,
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


# use_mask = False
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 11.51
# ===> barrier - IoU = 45.87
# ===> bicycle - IoU = 24.65
# ===> bus - IoU = 41.75
# ===> car - IoU = 46.15
# ===> construction_vehicle - IoU = 20.96
# ===> motorcycle - IoU = 26.82
# ===> pedestrian - IoU = 26.77
# ===> traffic_cone - IoU = 29.66
# ===> trailer - IoU = 24.65
# ===> truck - IoU = 32.75
# ===> driveable_surface - IoU = 60.39
# ===> other_flat - IoU = 32.87
# ===> sidewalk - IoU = 36.49
# ===> terrain - IoU = 33.16
# ===> manmade - IoU = 21.3
# ===> vegetation - IoU = 20.92
# ===> mIoU of 6019 samples: 31.57
# {'mIoU': array([0.115, 0.459, 0.247, 0.418, 0.461, 0.21 , 0.268, 0.268, 0.297,
#        0.247, 0.328, 0.604, 0.329, 0.365, 0.332, 0.213, 0.209, 0.839])}


# +----------------------+----------+----------+----------+
# |     Class Names      | RayIoU@1 | RayIoU@2 | RayIoU@4 |
# +----------------------+----------+----------+----------+
# |        others        |  0.101   |  0.108   |  0.110   |
# |       barrier        |  0.439   |  0.480   |  0.497   |
# |       bicycle        |  0.258   |  0.286   |  0.293   |
# |         bus          |  0.540   |  0.649   |  0.700   |
# |         car          |  0.531   |  0.603   |  0.629   |
# | construction_vehicle |  0.180   |  0.252   |  0.282   |
# |      motorcycle      |  0.247   |  0.328   |  0.343   |
# |      pedestrian      |  0.347   |  0.393   |  0.409   |
# |     traffic_cone     |  0.346   |  0.371   |  0.378   |
# |       trailer        |  0.209   |  0.292   |  0.384   |
# |        truck         |  0.452   |  0.544   |  0.587   |
# |  driveable_surface   |  0.562   |  0.646   |  0.734   |
# |      other_flat      |  0.290   |  0.328   |  0.363   |
# |       sidewalk       |  0.261   |  0.313   |  0.363   |
# |       terrain        |  0.260   |  0.330   |  0.394   |
# |       manmade        |  0.345   |  0.421   |  0.471   |
# |      vegetation      |  0.229   |  0.337   |  0.423   |
# +----------------------+----------+----------+----------+
# |         MEAN         |  0.329   |  0.393   |  0.433   |
# +----------------------+----------+----------+----------+
# 6019it [10:36,  9.46it/s]
# +----------------------+---------+---------+---------+
# |     Class Names      | RayPQ@1 | RayPQ@2 | RayPQ@4 |
# +----------------------+---------+---------+---------+
# |        others        |  0.026  |  0.032  |  0.033  |
# |       barrier        |  0.184  |  0.232  |  0.253  |
# |       bicycle        |  0.088  |  0.103  |  0.108  |
# |         bus          |  0.311  |  0.406  |  0.458  |
# |         car          |  0.300  |  0.380  |  0.403  |
# | construction_vehicle |  0.032  |  0.057  |  0.081  |
# |      motorcycle      |  0.114  |  0.156  |  0.169  |
# |      pedestrian      |  0.025  |  0.030  |  0.031  |
# |     traffic_cone     |  0.071  |  0.081  |  0.085  |
# |       trailer        |  0.049  |  0.077  |  0.088  |
# |        truck         |  0.182  |  0.274  |  0.314  |
# |  driveable_surface   |  0.457  |  0.574  |  0.702  |
# |      other_flat      |  0.062  |  0.086  |  0.106  |
# |       sidewalk       |  0.018  |  0.042  |  0.091  |
# |       terrain        |  0.017  |  0.039  |  0.074  |
# |       manmade        |  0.077  |  0.144  |  0.194  |
# |      vegetation      |  0.002  |  0.061  |  0.162  |
# +----------------------+---------+---------+---------+
# |         MEAN         |  0.119  |  0.163  |  0.197  |
# +----------------------+---------+---------+---------+
# {'RayIoU': 0.3850202377154096, 'RayIoU@1': 0.3291477679560127, 'RayIoU@2': 0.39307010079658805, 'RayIoU@4': 0.4328428443936281, 
#  'RayPQ': 0.15961266397677248, 'RayPQ@1': 0.11850092407498894, 'RayPQ@2': 0.1631862461686837, 'RayPQ@4': 0.19715082168664483}
