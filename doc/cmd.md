# Training cmd

## 1. FlashOcc
```shell script
bash tool/dist_train.sh projects/configs/flashocc/flashocc-r50-M0.py 4                             # 31.95
bash tool/dist_train.sh projects/configs/flashocc/flashocc-r50.py 4                                # 32.08
bash tool/dist_train.sh projects/configs/flashocc/flashocc-r50-4d-stereo.py 4                      # 37.84
bash tool/dist_train.sh projects/configs/flashocc/flashocc-stbase-4d-stereo-512x1408_4x4_1e-4.py 4 # 41.80
bash tool/dist_train.sh projects/configs/flashocc/flashocc-stbase-4d-stereo-512x1408_4x4_2e-4.py 4 # 43.52
```

## 2. Panoptic-FlashOcc
### for train
```shell script
conda activate FlashOcc
exp_name=panoptic-flashocc-r50-depth-tiny-pano
exp_name=panoptic-flashocc-r50-depth-pano
exp_name=panoptic-flashocc-r50-depth4d-pano
exp_name=panoptic-flashocc-r50-depth4d-longterm8f-pano
bash tools/dist_train.sh \
    projects/configs/panoptic-flashocc/${exp_name}.py \
    4
```

### for test
```shell script
conda activate FlashOcc
exp_name=panoptic-flashocc-r50-depth-tiny-pano
exp_name=panoptic-flashocc-r50-depth-pano
exp_name=panoptic-flashocc-r50-depth4d-pano
exp_name=panoptic-flashocc-r50-depth4d-longterm8f-pano
bash tools/dist_test.sh \
    projects/configs/panoptic-flashocc/${exp_name}.py \
    work_dirs/${exp_name}/epoch_24_ema.pth \
    4 \
    --eval ray-iou
```

### for vis
```shell script
exp_name=panoptic-flashocc-r50-depth-tiny-pano
exp_name=panoptic-flashocc-r50-depth-pano
exp_name=panoptic-flashocc-r50-depth4d-pano
exp_name=panoptic-flashocc-r50-depth4d-longterm8f-pano
python tools/vis_occ.py --config projects/configs/panoptic-flashocc/${exp_name}.py --weights work_dirs/${exp_name}/epoch_24_ema.pth --viz-dir vis/${exp_name} --draw-gt
```

### for test inference time
```shell script
conda activate FlashOcc
source activate FlashOcc
exp_name=panoptic-flashocc-r50-depth-tiny-pano
exp_name=panoptic-flashocc-r50-depth-pano
python tools/analysis_tools/benchmark.py \
    projects/configs/panoptic-flashocc/${exp_name}.py \
    work_dirs/${exp_name}/epoch_24_ema.pth \
    --w_pano --w_panoproc

exp_name=panoptic-flashocc-r50-depth4d-pano
exp_name=panoptic-flashocc-r50-depth4d-longterm8f-pano
python tools/analysis_tools/benchmark_sequential.py \
    projects/configs/panoptic-flashocc/${exp_name}.py \
    work_dirs/${exp_name}/epoch_24_ema.pth \
    --w_pano --w_panoproc

```
