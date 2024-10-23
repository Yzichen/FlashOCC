
# FlashOcc
```shell
# step 1. generate result 
bash tools/dist_test.sh projects/configs/flashocc/flashocc-r50.py ckpts/flashocc-r50-256x704.pth 4 --eval map --eval-options show_dir=work_dirs/flashocc_r50/results
# step 2. visualization
python tools/analysis_tools/vis_occ.py work_dirs/flashocc_r50/results/ --root_path ./data/nuscenes --save_path ./vis
```

# Panoptic-FlashOcc
```shell

exp_name=panoptic-flashocc-r50-depth4d-longterm8f-pano
python tools/vis_occ.py --config projects/configs/panoptic-flashocc/${exp_name}.py --weights work_dirs/${exp_name}/epoch_24_ema.pth --viz-dir vis/${exp_name} --draw-pano-gt

```

