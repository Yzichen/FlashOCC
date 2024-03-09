## Training cmd

1. FlashOcc
```shell script
bash tool/dist_train.sh projects/configs/flashocc/flashocc-r50-M0.py 4                             # 31.95
bash tool/dist_train.sh projects/configs/flashocc/flashocc-r50.py 4                                # 32.08
bash tool/dist_train.sh projects/configs/flashocc/flashocc-r50-4d-stereo.py 4                      # 37.84
bash tool/dist_train.sh projects/configs/flashocc/flashocc-stbase-4d-stereo-512x1408_4x4_1e-4.py 4 # 41.80
bash tool/dist_train.sh projects/configs/flashocc/flashocc-stbase-4d-stereo-512x1408_4x4_2e-4.py 4 # 43.52
```
