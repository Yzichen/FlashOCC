
#### Train model

conda activate FlashOcc

```shell
# step 1. generate result 
bash tools/dist_test.sh projects/configs/flashocc/flashocc-r50.py ckpts/flashocc-r50-256x704.pth 4 --eval map --eval-options show_dir=work_dirs/flashocc_r50/results
# step 2. visualization
python tools/analysis_tools/vis_occ.py work_dirs/flashocc_r50/results/ --root_path ./data/nuscenes --save_path ./vis
```

