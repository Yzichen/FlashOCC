_base_ = ['../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
          '../../../mmdetection3d/configs/_base_/default_runtime.py']

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
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
    type='BEVDepth4DOCC',
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
    )
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
                                'mask_lidar', 'mask_camera'])
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
            dict(type='Collect3D', keys=['points', 'img_inputs'])
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
# ===> others - IoU = 11.5
# ===> barrier - IoU = 44.1
# ===> bicycle - IoU = 25.89
# ===> bus - IoU = 41.0
# ===> car - IoU = 44.57
# ===> construction_vehicle - IoU = 21.88
# ===> motorcycle - IoU = 27.31
# ===> pedestrian - IoU = 25.95
# ===> traffic_cone - IoU = 29.04
# ===> trailer - IoU = 24.17
# ===> truck - IoU = 31.81
# ===> driveable_surface - IoU = 60.74
# ===> other_flat - IoU = 33.84
# ===> sidewalk - IoU = 36.62
# ===> terrain - IoU = 33.96
# ===> manmade - IoU = 21.54
# ===> vegetation - IoU = 21.36
# ===> mIoU of 6019 samples: 31.49
# {'mIoU': array([0.115, 0.441, 0.259, 0.41 , 0.446, 0.219, 0.273, 0.259, 0.29 ,
#        0.242, 0.318, 0.607, 0.338, 0.366, 0.34 , 0.215, 0.214, 0.839])}


# +----------------------+----------+----------+----------+
# |     Class Names      | RayIoU@1 | RayIoU@2 | RayIoU@4 |
# +----------------------+----------+----------+----------+
# |        others        |  0.107   |  0.115   |  0.116   |
# |       barrier        |  0.442   |  0.485   |  0.501   |
# |       bicycle        |  0.267   |  0.296   |  0.302   |
# |         bus          |  0.533   |  0.632   |  0.683   |
# |         car          |  0.516   |  0.590   |  0.616   |
# | construction_vehicle |  0.170   |  0.251   |  0.282   |
# |      motorcycle      |  0.231   |  0.325   |  0.350   |
# |      pedestrian      |  0.340   |  0.386   |  0.400   |
# |     traffic_cone     |  0.348   |  0.372   |  0.380   |
# |       trailer        |  0.232   |  0.317   |  0.400   |
# |        truck         |  0.427   |  0.514   |  0.559   |
# |  driveable_surface   |  0.566   |  0.649   |  0.736   |
# |      other_flat      |  0.302   |  0.341   |  0.374   |
# |       sidewalk       |  0.261   |  0.313   |  0.363   |
# |       terrain        |  0.258   |  0.333   |  0.399   |
# |       manmade        |  0.348   |  0.426   |  0.479   |
# |      vegetation      |  0.234   |  0.342   |  0.430   |
# +----------------------+----------+----------+----------+
# |         MEAN         |  0.328   |  0.393   |  0.434   |
# +----------------------+----------+----------+----------+
# {'RayIoU': 0.3851476341258822, 'RayIoU@1': 0.3284556495395326, 'RayIoU@2': 0.39334760720480005, 'RayIoU@4': 0.43363964563331386}