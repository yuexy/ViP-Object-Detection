# Visual Parser: Representing Part-whole Hierarchies with Transformers

This repository contains the official implementation to reproduce object detection results of [ViP](https://arxiv.org/abs/2107.05790). It is based on [mmdetection](https://github.com/open-mmlab/mmdetection).

## Results and Models

### Cascade Mask R-CNN
| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | #params | FLOPs | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| ViP-Ti | ImageNet-1K | 1x | 45.3 | 39.8 | 69.2M | 678G |[config](configs/vip/vip_t_cascade_mask_rcnn_1x.py) | [Google Drive](https://drive.google.com/file/d/19CYkLdQjnATukMcW7anVWc6h_rkn6U8G/view?usp=sharing) | [Google Drive](https://drive.google.com/file/d/1ha735PEjAaFbhpAY2LGKH1lJKKf0fDe1/view?usp=sharing) |
| ViP-S | ImageNet-1K | 1x | 48.0 | 42.0 | 87.1M | 725G |[config](configs/vip/vip_s_cascade_mask_rcnn_1x.py) | [Google Drive](https://drive.google.com/file/d/19MBkuSZXPYwzbdTBio4tM2rjZ4Y0gqHF/view?usp=sharing) | [Google Drive](https://drive.google.com/file/d/187Zw4eMx8q7suxxXjtH_hjfX_RRBN_zE/view?usp=sharing) |
| ViP-M | ImageNet-1K | 1x | 49.9 | 43.5 | 107.0M | 785G |- | - | Coming Soon |


### RetinaNet
| Backbone | Pretrain | Lr Schd | box mAP | #params | FLOPs | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| ViP-Ti | ImageNet-1k | 1x | 39.9 | 21.4M | 181G | [config](configs/vip/vip_t_retinanet_1x.py) | [Google Drive](https://drive.google.com/file/d/1sdBdT8O2CatVMvrXveQCuwfCpElVz_k-/view?usp=sharing) | [Google Drive](https://drive.google.com/file/d/1OA2lwmMtcYvoAtu_29nHSbQGQRuwIauP/view?usp=sharing) |
| ViP-S | ImageNet-1k | 1x | 42.7 | 39.9M | 227G | [config](configs/vip/vip_s_retinanet_1x.py) | [Google Drive](https://drive.google.com/file/d/1XBhXT6wM_HIJswfQeL0ypXp5Qzrqlj7-/view?usp=sharing) | [Google Drive](https://drive.google.com/file/d/1jq0t-9pM5n0uG_F11QnoDNSFad3nU446/view?usp=sharing) |
| ViP-S | ImageNet-1k | 3x | 43.9 | 39.9M | 227G | [config](configs/vip/vip_s_retinanet_3x.py) | [Google Drive](https://drive.google.com/file/d/1LlPFbix_qB88KtsG4kmAV-SrgB529ZK3/view?usp=sharing) | [Google Drive](https://drive.google.com/file/d/1XbX8gec9_CiT54_oWC0BRw79equ4jGEq/view?usp=sharing) |
|ViP-M | ImageNet-1k | 1x | 44.3 | 59.8M | 287G | - | - | Coming Soon |


**Notes**:

- **Pre-trained models can be downloaded from [Visual Parser](https://github.com/kevin-ssy/ViP)**.

## Usage

### Installation

Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md) for installation and dataset preparation.

### Inference
```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm
```

### Training

To train a detector with pre-trained models, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE>

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM>
```

## Citing ViP
```
@article{sun2021visual,
  title={Visual Parser: Representing Part-whole Hierarchies with Transformers},
  author={Sun, Shuyang and Yue, Xiaoyu, Bai, Song and Torr, Philip},
  journal={arXiv preprint arXiv:2107.05790},
  year={2021}
}
```
