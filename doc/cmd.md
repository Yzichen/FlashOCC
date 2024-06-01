## Training cmd

1. FlashOcc
```shell script
bash tool/dist_train.sh projects/configs/flashocc/flashocc-r50-M0.py 4                             # 31.95
bash tool/dist_train.sh projects/configs/flashocc/flashocc-r50.py 4                                # 32.08
bash tool/dist_train.sh projects/configs/flashocc/flashocc-r50-4d-stereo.py 4                      # 37.84
bash tool/dist_train.sh projects/configs/flashocc/flashocc-stbase-4d-stereo-512x1408_4x4_1e-4.py 4 # 41.80
bash tool/dist_train.sh projects/configs/flashocc/flashocc-stbase-4d-stereo-512x1408_4x4_2e-4.py 4 # 43.52
```


PORT=25001 CUDA_VISIBLE_DEVICES=6,7,4,5 bash tools/dist_test.sh projects/configs/flashoccv2/flashoccv2-r50-depth4d-longterm8f-pano.py work_dirs/flashoccv2-r50-depth4d-longterm8f-pano/epoch_24_ema.pth  4 --eval  ray-iou
PORT=25002 CUDA_VISIBLE_DEVICES=2,3,0,1 bash tools/dist_test.sh projects/configs/flashoccv2/flashoccv2-r50-depth4d-pano.py work_dirs/flashoccv2-r50-depth4d-pano/epoch_24_ema.pth  4 --eval  ray-iou

PORT=25001 CUDA_VISIBLE_DEVICES=6,7,4,5 bash tools/dist_test.sh projects/configs/flashoccv2/flashoccv2-r50-depth-pano.py work_dirs/flashoccv2-r50-depth-pano/epoch_24_ema.pth  4 --eval  ray-iou
PORT=25002 CUDA_VISIBLE_DEVICES=6,7,4,5 bash tools/dist_test.sh projects/configs/flashoccv2/flashoccv2-r50-depth-tiny-pano.py work_dirs/flashoccv2-r50-depth-tiny-pano/epoch_24_ema.pth  4 --eval  ray-iou


bash tools/dist_test.sh projects/configs/flashoccv2/flashoccv2-r50-depth4d-longterm8f-pano.py work_dirs/flashoccv2-r50-depth4d-longterm8f-pano/epoch_24_ema.pth 8 --eval miou --show-dir   work_dirs/flashoccv2-r50-depth4d-longterm8f-pano/results

exp_name=flashoccv2-r50-depth4d-pano
bash tools/dist_test.sh projects/configs/flashoccv2/${exp_name}.py work_dirs/${exp_name}/epoch_24_ema.pth 8 --eval map --eval-options show_dir=./work_dirs/${exp_name}/results
python tools/vis_occ_v2.py ./work_dirs/${exp_name}/results --viz-dir ./work_dirs/${exp_name}/vis --draw-gt --vis_frames 6019

exp_name=flashoccv2-r50-depth4d-longterm8f-pano
PORT=25002 bash tools/dist_test.sh projects/configs/flashoccv2/${exp_name}.py work_dirs/${exp_name}/epoch_24_ema.pth 8 --eval map --eval-options show_dir=./work_dirs/${exp_name}/results
python tools/vis_occ_v2.py ./work_dirs/${exp_name}/results --viz-dir ./work_dirs/${exp_name}/vis --draw-gt --vis_frames 6019


bash tools/dist_test.sh projects/configs/flashoccv2/flashoccv2-r50-depth4d.py work_dirs/flashoccv2-r50-depth4d/epoch_24_ema.pth 8 --eval map --eval-options show_dir=./work_dirs/flashoccv2-r50-depth4d/results
python tools/vis_occ_v2.py ./work_dirs/flashoccv2-r50-depth4d/results --viz-dir ./work_dirs/flashoccv2-r50-depth4d/vis --draw-gt --vis_frames 6019