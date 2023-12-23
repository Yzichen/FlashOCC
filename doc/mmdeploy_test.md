# trt inference
```shell
conda activate FlashOcc
exp_name=flashocc-r50
fold_name=flashocc
config=projects/configs/${fold_name}/${exp_name}-trt.py
checkpoint=ckpts/flashocc-r50-256x704.pth
work_dir=work_dirs/${exp_name}/onnx_trt/


# int8 test. 
# # Warning! This is a simple ptq setting, it would drop the performance. We'll fix it later, but you can use it to test other performance.
engine=work_dirs/${exp_name}/onnx_trt/bevdet_int8_fuse.engine
CUDA_VISIBLE_DEVICES=0 python tools/convert_bevdet_to_TRT.py $config $checkpoint $work_dir --fuse-conv-bn --fp16 --int8 --calib_num 32
python tools/analysis_tools/benchmark_trt.py $config $engine --eval

# fp16 test
engine=work_dirs/${exp_name}/onnx_trt/bevdet_fp16_fuse.engine
python tools/convert_bevdet_to_TRT.py $config $checkpoint $work_dir --fuse-conv-bn --fp16
python tools/analysis_tools/benchmark_trt.py $config $engine
python tools/analysis_tools/benchmark_trt.py $config $engine --eval
# # result
    # ===> per class IoU of 6019 samples:
    # ===> others - IoU = 6.13
    # ===> barrier - IoU = 34.61
    # ===> bicycle - IoU = 10.18
    # ===> bus - IoU = 38.18
    # ===> car - IoU = 41.48
    # ===> construction_vehicle - IoU = 14.27
    # ===> motorcycle - IoU = 14.07
    # ===> pedestrian - IoU = 14.73
    # ===> traffic_cone - IoU = 13.38
    # ===> trailer - IoU = 26.69
    # ===> truck - IoU = 29.85
    # ===> driveable_surface - IoU = 76.88
    # ===> other_flat - IoU = 34.85
    # ===> sidewalk - IoU = 46.85
    # ===> terrain - IoU = 51.72
    # ===> manmade - IoU = 37.08
    # ===> vegetation - IoU = 32.04
    # ===> mIoU of 6019 samples: 30.76

# fp32 test
engine=work_dirs/${exp_name}/onnx_trt/bevdet_fuse.engine
python tools/convert_bevdet_to_TRT.py $config $checkpoint $work_dir --fuse-conv-bn
python tools/analysis_tools/benchmark_trt.py $config $engine
python tools/analysis_tools/benchmark_trt.py $config $engine --eval
```
