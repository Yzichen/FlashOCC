import os
import cv2
import logging
import argparse
import importlib
import torch
import numpy as np
from tqdm import tqdm
from mmcv import Config, DictAction
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset, build_dataloader
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
import mmdet
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

import sys
sys.path.insert(0, os.getcwd())

if mmdet.__version__ > '2.23.0':
    # If mmdet version > 2.23.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

color_map = np.array([
    [0, 0, 0, 255],    # others
    [255, 120, 50, 255],  # barrier              orangey
    [255, 192, 203, 255],  # bicycle              pink
    [255, 255, 0, 255],  # bus                  yellow
    [0, 150, 245, 255],  # car                  blue
    [0, 255, 255, 255],  # construction_vehicle cyan
    [200, 180, 0, 255],  # motorcycle           dark orange
    [255, 0, 0, 255],  # pedestrian           red
    [255, 240, 150, 255],  # traffic_cone         light yellow
    [135, 60, 0, 255],  # trailer              brown
    [160, 32, 240, 255],  # truck                purple
    [255, 0, 255, 255],  # driveable_surface    dark pink
    [175,   0,  75, 255],       # other_flat           dark red
    [75, 0, 75, 255],  # sidewalk             dard purple
    [150, 240, 80, 255],  # terrain              light green
    [230, 230, 250, 255],  # manmade              white
    [0, 175, 0, 255],  # vegetation           green
    [255, 255, 255, 255],  # free             white
], dtype=np.uint8)

# # from matplotlib import colors
# # hex_code_list = [
# #     '#000000', '#D3D3D3', '#BC8F8F', '#F08080', '#A52A2A', '#FF0000', '#FFA07A', '#A0522D', '#FFE4C4', '#FFE4B5',  \
# #     '#DAA520', '#FFD700', '#F0E68C', '#BDB76B', '#808000', '#FFFF00', '#9ACD32', '#7FFF00', '#8FBC8F', '#90EE90',  \
# #     '#32CD32', '#008000', '#00FF00', '#00FA9A', '#7FFFD4', '#48D1CC', '#2F4F4F', '#ADD8E6', '#87CEFA', '#DC143C',  \
# #     '#696969', '#9370DB', '#8A2BE2', '#9400D3', '#DDA0DD', '#FF00FF', '#C71585', '#DB7093', '#FFB6C1', '#bf9b0c',  \
# #     '#01889f', '#bb3f3f', '#1805db', '#48c072', '#fffd37', '#c44240', '#6140ef', '#ceaefa', '#04f489', '#c6f808',  \
# #     '#507b9c', '#cffdbc', '#ac7e04', '#01386a', '#ffb7ce', '#ffd1df', '#D2691E', '#FFDAB9', '#a55af4', '#95d0fc',  \
# #     ]
# # hex_code_list = np.array(hex_code_list).reshape(6,10).transpose(1,0).reshape(-1)
# # pano_color_map = np.array([[int(value * 255) for value in colors.hex2color(hex_code)] for hex_code in hex_code_list], dtype=np.uint8)

import matplotlib.pyplot as plt
from scipy.ndimage import rotate
def draw_fig(tensor, name='tensor_image_colored_no_white.png'):
    tensor = tensor.squeeze(0)
    tensor = rotate(tensor, -90, reshape=False)
    tensor = np.flip(tensor, axis=1)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(tensor, cmap='viridis')
    plt.axis('off')
    plt.savefig(name, bbox_inches='tight', pad_inches=0)
    plt.clf()
    
def generate_rgb_color(number):
    red = (number % 256)
    green = ((number // 256) % 256)
    blue = ((number // 65536) % 256)
    return [red, green, blue]
pano_color_map = np.array([generate_rgb_color(number) for number in np.random.randint(0, 65536*256, 256)])

inst_class_ids=[2, 3, 4, 5, 6, 7, 9, 10]

def occ2img(semantics=None, is_pano=False, panoptics=None):
    H, W, D = semantics.shape

    free_id = len(occ_class_names) - 1
    semantics_2d = np.ones([H, W], dtype=np.int32) * free_id
    for i in range(D):
        semantics_i = semantics[..., i]
        non_free_mask = (semantics_i != free_id)
        semantics_2d[non_free_mask] = semantics_i[non_free_mask]

    viz = color_map[semantics_2d]
    viz = viz[..., :3]

    inst_mask = np.zeros_like(semantics_2d).astype(np.bool)
    for ind in inst_class_ids:
        inst_mask[semantics_2d==ind] = True
    
    if is_pano:
        panoptics_2d = np.ones([H, W], dtype=np.int32) * 0
        for i in range(D):
            panoptics_i = panoptics[..., i]
            semantics_i = semantics[..., i]
            non_free_mask = (semantics_i != free_id)
            panoptics_2d[non_free_mask] = panoptics_i[non_free_mask]
        
        
        # # panoptics_2d = panoptics_2d%60
        
        
        viz_pano = pano_color_map[panoptics_2d]
        viz[inst_mask,:] = viz_pano[inst_mask,:]

    viz = cv2.resize(viz, dsize=(800, 800))
    return viz

def main():
    parser = argparse.ArgumentParser(description='Validate a detector')
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--viz-dir', required=True)
    parser.add_argument('--override', nargs='+', action=DictAction)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--draw-sem-gt', action='store_true')
    parser.add_argument('--draw-pano-gt', action='store_true')
    parser.add_argument('--surround-view-img', action='store_true')
    parser.add_argument('--surround-pano-gt', action='store_true')
    args = parser.parse_args()

    # parse configs
    cfgs = Config.fromfile(args.config)
    if args.override is not None:
        cfgs.merge_from_dict(args.override)

    cfgs = compat_cfg(cfgs)

    # set multi-process settings
    setup_multi_processes(cfgs)
    
    # import modules from plguin/xx, registry will be updated
    if hasattr(cfgs, 'plugin'):
        if cfgs.plugin:
            import importlib
            if hasattr(cfgs, 'plugin_dir'):
                plugin_dir = cfgs.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                plg_lib = importlib.import_module(_module_path)

    # use val-mini for visualization
    #cfgs.data.val.ann_file = cfgs.data.val.ann_file.replace('val', 'val_mini')

    # MMCV, please shut up
    from mmcv.utils.logging import logger_initialized
    logger_initialized['root'] = logging.Logger(__name__, logging.WARNING)
    logger_initialized['mmcv'] = logging.Logger(__name__, logging.WARNING)

    # you need one GPU
    assert torch.cuda.is_available()
    # assert torch.cuda.device_count() == 1

    # logging
    logging.info('Using GPU: %s' % torch.cuda.get_device_name(0))

    # random seed
    logging.info('Setting random seed: 0')
    set_random_seed(0, deterministic=True)

    logging.info('Loading validation set from %s' % cfgs.data.val.data_root)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfgs.dist_params)
        
    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)
    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfgs.data.get('test_dataloader', {})
    }
    dataset = build_dataset(cfgs.data.test)
    test_loader_cfg['workers_per_gpu'] = 2
    val_loader = build_dataloader(dataset, **test_loader_cfg)
    # val_dataset = build_dataset(cfgs.data.test)
    # val_loader = build_dataloader(
    #     val_dataset,
    #     samples_per_gpu=1,
    #     workers_per_gpu=1,
    #     num_gpus=1,
    #     dist=False,
    #     shuffle=False,
    #     seed=0,
    # )

    logging.info('Creating model: %s' % cfgs.model.type)
    model = build_model(cfgs.model)
    model.cuda()
    model = MMDataParallel(model, [0])
    model.eval()

    logging.info('Loading checkpoint from %s' % args.weights)
    load_checkpoint(
        model, args.weights, map_location='cuda', strict=True,
        logger=logging.Logger(__name__, logging.ERROR)
    )

    if not os.path.exists(args.viz_dir):
        os.makedirs(args.viz_dir)
        
    for i, data in tqdm(enumerate(val_loader)):

        with torch.no_grad():
            occ_pred = model(return_loss=False, rescale=True, **data)[0]

            if False:
                occ_bev_feature = occ_pred['occ_bev_feature']
                outs = occ_pred['outs']

                tensor = occ_bev_feature.max(dim=1)[0].cpu()
                draw_fig(tensor, name=os.path.join(args.viz_dir, '%04d-occ_bev_feature.jpg' % i))
                print(os.path.join(args.viz_dir, '%04d-occ_bev_feature.jpg' % i))

                tensor = outs[0][0]['heatmap'].sigmoid().sum(dim=1)[0].cpu()
                draw_fig(tensor, name=os.path.join(args.viz_dir, '%04d-heatmap.jpg' % i))
                print(os.path.join(args.viz_dir, '%04d-heatmap.jpg' % i))

                tensor = outs[0][0]['reg'][0,0].cpu()
                tensor = outs[0][0]['reg'][0,1].cpu()
                tensor = ((outs[0][0]['reg'][0,0]**2+outs[0][0]['reg'][0,1]**2)**0.5).unsqueeze(dim=0).cpu()
                draw_fig(tensor, name=os.path.join(args.viz_dir, '%04d-reg.jpg' % i))
                print(os.path.join(args.viz_dir, '%04d-reg.jpg' % i))

                tensor = outs[0][0]['height'][0,0].cpu()
                draw_fig(tensor, name=os.path.join(args.viz_dir, '%04d-height.jpg' % i))
                print(os.path.join(args.viz_dir, '%04d-height.jpg' % i))


            sem_pred = occ_pred['pred_occ']
            cv2.imwrite(os.path.join(args.viz_dir, '%04d-sem.jpg' % i), occ2img(semantics=sem_pred.cpu())[..., ::-1])
            print(os.path.join(args.viz_dir, '%04d-sem.jpg' % i))
            
            inst_pred = occ_pred['pano_inst']
            cv2.imwrite(os.path.join(args.viz_dir, '%04d-inst.jpg' % i), occ2img(semantics=sem_pred.cpu(), is_pano=True, panoptics=inst_pred.cpu())[..., ::-1])
            print(os.path.join(args.viz_dir, '%04d-inst.jpg' % i))
            
            if args.surround_view_img:
                img = data['img_inputs'][0][0][0][::9].cpu().numpy()
                mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1,3,1,1)
                std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1,3,1,1)
                img = img*std + mean
                img = img.astype(np.uint8).transpose(0,2,3,1)
                up = np.concatenate([img[0,...], img[1,...], img[2,...]], 1)
                down = np.concatenate([img[3,...], img[4,...], img[5,...]], 1)
                out = np.concatenate([up, down], 0)
                cv2.imwrite(os.path.join(args.viz_dir, '%04d-rgb.jpg' % i), out)
                print(os.path.join(args.viz_dir, '%04d-rgb.jpg' % i))

            if args.draw_sem_gt or args.draw_pano_gt:
                occ_gt = np.load(os.path.join(val_loader.dataset.data_infos[i]['occ_path'].\
                    replace('data/nuscenes/gts/', 'data/nuscenes/occ3d_panoptic/'), 'labels.npz'))
                pano_gt = occ_gt['instances']
                sem_gt = occ_gt['semantics']

            if args.draw_sem_gt:
                # sem_gt = np.array(data['voxel_semantics'][0])[0]
                cv2.imwrite(os.path.join(args.viz_dir, '%04d-sem-gt.jpg' % i), occ2img(semantics=sem_gt.cpu())[..., ::-1])

            if args.draw_pano_gt:
                cv2.imwrite(os.path.join(args.viz_dir, '%04d-pano-gt.jpg' % i), occ2img(semantics=sem_gt, is_pano=True, panoptics=pano_gt)[..., ::-1])
                print(os.path.join(args.viz_dir, '%04d-pano-gt.jpg' % i))

if __name__ == '__main__':
    main()

'''
exp_name=flashoccv2-r50-depth-tiny-pano
python tools/vis_occ.py --config projects/configs/flashoccv2/${exp_name}.py --weights work_dirs/${exp_name}/epoch_24_ema.pth --viz-dir vis/${exp_name} --draw-gt

exp_name=flashoccv2-r50-depth4d-longterm8f-pano
python tools/vis_occ.py --config projects/configs/flashoccv2/${exp_name}.py --weights work_dirs/${exp_name}/epoch_24_ema.pth --viz-dir vis/${exp_name} --draw-pano-gt #--draw-gt
'''