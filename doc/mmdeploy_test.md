# trt inference speed
```shell
conda activate FlashOcc
1. cmd for M0
exp_name=flashocc-r50-M0
fold_name=flashocc
config=projects/configs/${fold_name}/${exp_name}-trt.py
checkpoint=ckpts/flashocc-r50-M0-256x704.pth
work_dir=work_dirs/${exp_name}/onnx_trt/

2. cmd for M1
exp_name=flashocc-r50
fold_name=flashocc
config=projects/configs/${fold_name}/${exp_name}-trt.py
checkpoint=ckpts/flashocc-r50-256x704.pth
work_dir=work_dirs/${exp_name}/onnx_trt/


# int8 test. 
engine=work_dirs/${exp_name}/onnx_trt/bevdet_int8_fuse.engine
python tools/convert_bevdet_to_TRT.py $config $checkpoint $work_dir --fuse-conv-bn --int8 --calib_num 256
python tools/analysis_tools/benchmark_trt.py $config $engine --eval
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 4.58
# ===> barrier - IoU = 34.13
# ===> bicycle - IoU = 8.68
# ===> bus - IoU = 34.9
# ===> car - IoU = 40.48
# ===> construction_vehicle - IoU = 15.99
# ===> motorcycle - IoU = 15.49
# ===> pedestrian - IoU = 13.58
# ===> traffic_cone - IoU = 12.83
# ===> trailer - IoU = 25.31
# ===> truck - IoU = 28.08
# ===> driveable_surface - IoU = 76.7
# ===> other_flat - IoU = 31.5
# ===> sidewalk - IoU = 45.01
# ===> terrain - IoU = 49.63
# ===> manmade - IoU = 35.72
# ===> vegetation - IoU = 30.39
# ===> mIoU of 6019 samples: 29.59

# int8+fp16 test. 
engine=work_dirs/${exp_name}/onnx_trt/bevdet_int8_fp16_fuse.engine
python tools/convert_bevdet_to_TRT.py $config $checkpoint $work_dir --fuse-conv-bn --fp16 --int8 --calib_num 256
python tools/analysis_tools/benchmark_trt.py $config $engine --eval
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 4.59
# ===> barrier - IoU = 34.13
# ===> bicycle - IoU = 8.71
# ===> bus - IoU = 34.9
# ===> car - IoU = 40.49
# ===> construction_vehicle - IoU = 16.01
# ===> motorcycle - IoU = 15.55
# ===> pedestrian - IoU = 13.63
# ===> traffic_cone - IoU = 12.86
# ===> trailer - IoU = 25.33
# ===> truck - IoU = 28.1
# ===> driveable_surface - IoU = 76.7
# ===> other_flat - IoU = 31.51
# ===> sidewalk - IoU = 45.01
# ===> terrain - IoU = 49.63
# ===> manmade - IoU = 35.72
# ===> vegetation - IoU = 30.39
# ===> mIoU of 6019 samples: 29.6

# fp16 test
engine=work_dirs/${exp_name}/onnx_trt/bevdet_fp16_fuse.engine
python tools/convert_bevdet_to_TRT.py $config $checkpoint $work_dir --fuse-conv-bn --fp16
python tools/analysis_tools/benchmark_trt.py $config $engine
python tools/analysis_tools/benchmark_trt.py $config $engine --eval
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 5.97
# ===> barrier - IoU = 36.37
# ===> bicycle - IoU = 10.14
# ===> bus - IoU = 35.47
# ===> car - IoU = 41.57
# ===> construction_vehicle - IoU = 15.73
# ===> motorcycle - IoU = 14.8
# ===> pedestrian - IoU = 15.65
# ===> traffic_cone - IoU = 14.46
# ===> trailer - IoU = 27.47
# ===> truck - IoU = 29.39
# ===> driveable_surface - IoU = 77.14
# ===> other_flat - IoU = 34.66
# ===> sidewalk - IoU = 46.44
# ===> terrain - IoU = 51.05
# ===> manmade - IoU = 35.79
# ===> vegetation - IoU = 31.19
# ===> mIoU of 6019 samples: 30.78

# fp32 test
engine=work_dirs/${exp_name}/onnx_trt/bevdet_fuse.engine
python tools/convert_bevdet_to_TRT.py $config $checkpoint $work_dir --fuse-conv-bn
python tools/analysis_tools/benchmark_trt.py $config $engine
python tools/analysis_tools/benchmark_trt.py $config $engine --eval
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 5.97
# ===> barrier - IoU = 36.37
# ===> bicycle - IoU = 10.15
# ===> bus - IoU = 35.46
# ===> car - IoU = 41.56
# ===> construction_vehicle - IoU = 15.73
# ===> motorcycle - IoU = 14.78
# ===> pedestrian - IoU = 15.64
# ===> traffic_cone - IoU = 14.44
# ===> trailer - IoU = 27.46
# ===> truck - IoU = 29.39
# ===> driveable_surface - IoU = 77.14
# ===> other_flat - IoU = 34.68
# ===> sidewalk - IoU = 46.44
# ===> terrain - IoU = 51.05
# ===> manmade - IoU = 35.79
# ===> vegetation - IoU = 31.18
# ===> mIoU of 6019 samples: 30.78

```


3. cmd for flashoccv2
```
exp_name=flashoccv2-r50-depth
fold_name=flashoccv2
config=projects/configs/${fold_name}/${exp_name}-trt.py
checkpoint=work_dirs/${exp_name}/epoch_24_ema.pth
work_dir=work_dirs/${exp_name}/onnx_trt/

# fp16 test
engine=work_dirs/${exp_name}/onnx_trt/bevdet_fp16_fuse.engine
python tools/convert_bevdet_to_TRT.py $config $checkpoint $work_dir --fuse-conv-bn --fp16
python tools/analysis_tools/benchmark_trt.py $config $engine
python tools/analysis_tools/benchmark_trt.py $config $engine --eval
```

# Flops and params
```shell
python tools/analysis_tools/get_flops.py projects/configs/bevdet_occ/bevdet-occ-r50.py --modality image --shape 256 704
python tools/analysis_tools/get_flops.py projects/configs/flashocc/flashocc-r50-M0.py --modality image --shape 256 704
python tools/analysis_tools/get_flops.py projects/configs/flashocc/flashocc-r50.py --modality image --shape 256 704
python tools/analysis_tools/get_flops.py projects/configs/flashocc/flashocc-stbase-4d-stereo-512x1408.py --modality image --shape 512 1408
python tools/analysis_tools/get_flops.py projects/configs/flashoccv2/flashoccv2-r50-depth.py --modality image --shape 256 704
python tools/analysis_tools/get_flops.py projects/configs/flashoccv2/flashoccv2-r50-depth-tiny.py --modality image --shape 256 704
```