# FlashOcc: Fast and Memory-Efficient Occupancy Prediction via Channel-to-Height Plugin
<!-- ## Introduction -->

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2311.12058)

This repository is an official implementation of [FlashOCC](https://arxiv.org/abs/2311.12058) 

<div align="center">
  <img src="figs/overview.png"/>
</div><br/>
Given the capability of mitigating the long-tail deficiencies and intricate-shaped absence prevalent in 3D object detection, occupancy prediction 
has become a pivotal component in autonomous driving systems. However, the procession of three-dimensional voxel-level representations inevitably 
introduces large overhead in both memory and computation, obstructing the deployment of to-date occupancy prediction approaches. In contrast to the 
trend of making the model larger and more complicated, we argue that a desirable framework should be deployment-friendly to diverse chips while 
maintaining high precision. To this end, we propose a plug-and-play paradigm, namely FlashOCC, to consolidate rapid and memory-efficient occupancy 
prediction while maintaining high precision. Particularly, our FlashOCC makes two improvements based on the contemporary voxel-level occupancy prediction 
approaches. Firstly, the features are kept in the BEV, enabling the employment of efficient 2D convolutional layers for feature extraction. Secondly, 
a channel-to-height transformation is introduced to lift the output logits from the BEV into the 3D space. We apply the FlashOCC to diverse occupancy 
prediction baselines on the challenging Occ3D-nuScenes benchmarks and conduct extensive experiments to validate the effectiveness. The results substantiate 
the superiority of our plug-and-play paradigm over previous state-of-the-art methods in terms of precision, runtime efficiency, and memory costs, 
demonstrating its potential for deployment.


## Main Results
### Nuscenes Occupancy
| Config                                                                                                    | mIOU  | Model                                                             | Log                                                                                          |
|-----------------------------------------------------------------------------------------------------------|-------|-------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| [**FlashOCC-R50**](projects/configs/flashocc/flashocc-r50.py)                                             | 32.08 | [gdrive](https://drive.google.com/file/d/1k9BzXB2nRyvXhqf7GQx3XNSej6Oq6I-B/view?usp=drive_link) | [log](https://drive.google.com/file/d/1NRm27wVZMSUylmZxsMedFSLr7729YEAV/view?usp=drive_link) |
| [**FlashOCC-R50-4D-Stereo**](projects/configs/flashocc/flashocc-r50-4d-stereo.py)                         | 37.84 | [gdrive](https://drive.google.com/file/d/12WYaCdoZA8-A6_oh6vdLgOmqyEc3PNCe/view?usp=drive_link) | [log](https://drive.google.com/file/d/1eYvu9gUSQ7qk7w7lWPLrZMB0O2uKQUk3/view?usp=drive_link) |
| [**FlashOCC-STBase-4D-Stereo-512x1408**](projects/configs/flashocc/flashocc-stbase-4d-stereo-512x1408.py) | 43.52 | [gdrive](https://drive.google.com/file/d/1f6E6Bm6enIJETSEbfXs57M0iOUU997kU/view?usp=drive_link) | [log](https://drive.google.com/file/d/1tch-YK4ROGDGNmDcN5FZnOAvsbHe-iSU/view?usp=drive_link) |

## Get Started
step 1. Please prepare environment as that in [Docker](docker/Dockerfile).

step 2. Prepare flashocc repo by.
```shell script
git clone https://github.com/YZichen/FlashOCC.git
cd FlashOCC/projects
pip install -v -e .
```

step 3. Prepare nuScenes dataset as introduced in [nuscenes_det.md](docs/en/datasets/nuscenes_det.md) and create the pkl for FlashOCC by running:
```shell
python tools/create_data_bevdet.py
```
step 4. For Occupancy Prediction task, download (only) the 'gts' from [CVPR2023-3D-Occupancy-Prediction](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction) and arrange the folder as:
```shell script
└── nuscenes
    ├── v1.0-trainval (existing)
    ├── sweeps  (existing)
    ├── samples (existing)
    └── gts (new)
```

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
```



## Acknowledgement
Many thanks to the authors of [BEVDet](https://github.com/HuangJunJie2017/BEVDet), and the main code is based on it.

## Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@article{yu2023flashocc,
      title={FlashOcc: Fast and Memory-Efficient Occupancy Prediction via Channel-to-Height Plugin}, 
      author={Zichen Yu and Changyong Shu and Jiajun Deng and Kangjie Lu and Zongdai Liu and Jiangyong Yu and Dawei Yang and Hui Li and Yan Chen},
      year={2023},
      eprint={2311.12058},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```
