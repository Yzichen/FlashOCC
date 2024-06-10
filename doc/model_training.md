
#### Train model
```shell
# single gpu
python tools/train.py $config
# multiple gpu
./tools/dist_train.sh $config num_gpu
```

#### Test model
```shell
# single gpu
python tools/test.py $config $checkpoint --eval mAP
# multiple gpu
./tools/dist_test.sh $config $checkpoint num_gpu --eval mAP
# ray-iou metric
./tools/dist_test.sh $config $checkpoint num_gpu --eval ray-iou
```

#### FPS for Panoptic-FlashOcc
```shell
# for single-frame
python tools/analysis_tools/benchmark.py  config ckpt 
python tools/analysis_tools/benchmark.py  config ckpt --w_pano

# for multi-frame
python tools/analysis_tools/benchmark_sequential.py  config ckpt 
python tools/analysis_tools/benchmark_sequential.py  config ckpt --w_pano
```
