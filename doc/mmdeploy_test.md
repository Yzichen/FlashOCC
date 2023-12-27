# trt inference
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
# # Warning! This is a simple ptq setting, it would drop the performance. We'll fix it later, but you can use it to test other performance.
engine=work_dirs/${exp_name}/onnx_trt/bevdet_int8_fuse.engine
CUDA_VISIBLE_DEVICES=0 python tools/convert_bevdet_to_TRT.py $config $checkpoint $work_dir --fuse-conv-bn --fp16 --int8 --calib_num 32
python tools/analysis_tools/benchmark_trt.py $config $engine --eval

# fp16 test
engine=work_dirs/${exp_name}/onnx_trt/bevdet_fp16_fuse.engine
python tools/convert_bevdet_to_TRT.py $config $checkpoint $work_dir --fuse-conv-bn --fp16
python tools/analysis_tools/benchmark_trt.py $config $engine
python tools/analysis_tools/benchmark_trt.py $config $engine --eval

# fp32 test
engine=work_dirs/${exp_name}/onnx_trt/bevdet_fuse.engine
python tools/convert_bevdet_to_TRT.py $config $checkpoint $work_dir --fuse-conv-bn
python tools/analysis_tools/benchmark_trt.py $config $engine
python tools/analysis_tools/benchmark_trt.py $config $engine --eval
```

