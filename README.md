# FlashOcc: Fast and Memory-Efficient Occupancy Prediction via Channel-to-Height Plugin

## News
- **2023.12.23** Release the quick testing code via TensorRT in MMDeploy.
- **2023.11.28** Release the training code for FlashOcc.

<!-- - [History](./docs/en/news.md) -->

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
| Config                                                                                                    | mIOU  | FPS  | Model                                                             | Log                                                                                          |
|-----------------------------------------------------------------------------------------------------------|-------|-------|-------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| [**BEVDetOCC-R50**](projects/configs/bevdet_occ/bevdet-occ-r50.py)                                        | 31.60 | 92.1 | [gdrive]() | [log]() |
| [**M0:FO(BEVDetOCC)-R50**](projects/configs/flashocc/flashocc-r50.py)                                        | 31.95 | [197.6](doc/mmdeploy_test.md) | [gdrive](https://drive.google.com/file/d/14my3jdqiIv6VIrkozQ6-ruEcBOPVlWGJ/view?usp=sharing) | [log](https://drive.google.com/file/d/1E-kaHxbTr6s3Qn70vhKpwJM8kejoNFxQ/view?usp=sharing) |
| [**M1:FO(BEVDetOCC)-R50**](projects/configs/flashocc/flashocc-r50.py)                                        | 32.08 | [152.7](doc/mmdeploy_test.md) | [gdrive](https://drive.google.com/file/d/1k9BzXB2nRyvXhqf7GQx3XNSej6Oq6I-B/view?usp=drive_link) | [log](https://drive.google.com/file/d/1NRm27wVZMSUylmZxsMedFSLr7729YEAV/view?usp=drive_link) |
| [**BEVDetOCC-R50-4D-Stereo**](projects/configs/bevdet_occ/bevdet-occ-r50-4d-stereo.py)                    | 36.1 | - | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) | [log](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) |
| [**M2:FO(BEVDetOCC)-R50-4D-Stereo**](projects/configs/flashocc/flashocc-r50-4d-stereo.py)                         | 37.84 | - | [gdrive](https://drive.google.com/file/d/12WYaCdoZA8-A6_oh6vdLgOmqyEc3PNCe/view?usp=drive_link) | [log](https://drive.google.com/file/d/1eYvu9gUSQ7qk7w7lWPLrZMB0O2uKQUk3/view?usp=drive_link) |
| [**BEVDetOCC-STBase-4D-Stereo-512x1408**](projects/configs/bevdet_occ/bevdet-occ-stbase-4d-stereo-512x1408.py) | 42.0 | - | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) | [log](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) |
| [**M3:FO(BEVDetOCC)-STBase-4D-Stereo-512x1408**](projects/configs/flashocc/flashocc-stbase-4d-stereo-512x1408.py) | 43.52 | - | [gdrive](https://drive.google.com/file/d/1f6E6Bm6enIJETSEbfXs57M0iOUU997kU/view?usp=drive_link) | [log](https://drive.google.com/file/d/1tch-YK4ROGDGNmDcN5FZnOAvsbHe-iSU/view?usp=drive_link) |

FPS are tested via TensorRT on 3090 with FP16 precision. Please refer to Tab.2 in paper for the detail model settings for M-number.

## Get Started
1. [Environment Setup](doc/install.md)
2. [Model Training](doc/model_training.md)
3. [Quick Test Via TensorRT In MMDeploy](doc/mmdeploy_test.md)
4. [Visualization](doc/visualization.md)
5. [TensorRT Implement Writen In C++ With Cuda Acceleration](), will come soon


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
