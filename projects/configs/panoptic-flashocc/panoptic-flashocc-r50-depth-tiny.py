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
    'depth': [1.0, 45.0, 1.0],
}

voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 64

model = dict(
    type='BEVDepthOCC',     # single-frame
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
                                'mask_lidar', 'mask_camera'])
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


# use_mask = False
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 10.69
# ===> barrier - IoU = 39.67
# ===> bicycle - IoU = 22.01
# ===> bus - IoU = 39.99
# ===> car - IoU = 40.46
# ===> construction_vehicle - IoU = 20.44
# ===> motorcycle - IoU = 24.52
# ===> pedestrian - IoU = 22.5
# ===> traffic_cone - IoU = 23.72
# ===> trailer - IoU = 25.93
# ===> truck - IoU = 29.75
# ===> driveable_surface - IoU = 58.29
# ===> other_flat - IoU = 31.46
# ===> sidewalk - IoU = 33.92
# ===> terrain - IoU = 31.25
# ===> manmade - IoU = 17.46
# ===> vegetation - IoU = 17.97
# ===> mIoU of 6019 samples: 28.83
# {'mIoU': array([0.1068576 , 0.3967071 , 0.220114  , 0.3998965 , 0.40462457,
#        0.20442682, 0.24516316, 0.22497209, 0.23719173, 0.25925541,
#        0.29754347, 0.58293305, 0.31458314, 0.33921965, 0.31254221,
#        0.17456574, 0.17970859, 0.8315865 ])}


# Starting Evaluation...
# 6019it [10:23,  9.65it/s]
# +----------------------+----------+----------+----------+
# |     Class Names      | RayIoU@1 | RayIoU@2 | RayIoU@4 |
# +----------------------+----------+----------+----------+
# |        others        |  0.094   |  0.107   |  0.111   |
# |       barrier        |  0.367   |  0.421   |  0.443   |
# |       bicycle        |  0.209   |  0.251   |  0.261   |
# |         bus          |  0.498   |  0.601   |  0.665   |
# |         car          |  0.472   |  0.550   |  0.581   |
# | construction_vehicle |  0.175   |  0.251   |  0.287   |
# |      motorcycle      |  0.205   |  0.292   |  0.315   |
# |      pedestrian      |  0.289   |  0.339   |  0.354   |
# |     traffic_cone     |  0.276   |  0.302   |  0.314   |
# |       trailer        |  0.203   |  0.289   |  0.380   |
# |        truck         |  0.396   |  0.493   |  0.546   |
# |  driveable_surface   |  0.528   |  0.611   |  0.702   |
# |      other_flat      |  0.280   |  0.315   |  0.346   |
# |       sidewalk       |  0.233   |  0.279   |  0.328   |
# |       terrain        |  0.218   |  0.286   |  0.353   |
# |       manmade        |  0.268   |  0.347   |  0.398   |
# |      vegetation      |  0.174   |  0.272   |  0.358   |
# +----------------------+----------+----------+----------+
# |         MEAN         |  0.287   |  0.353   |  0.397   |
# +----------------------+----------+----------+----------+
# {'RayIoU': 0.34574739050176573, 'RayIoU@1': 0.2873820616941079, 'RayIoU@2': 0.3533573712072785,
# 'RayIoU@4': 0.39650273860391083}
