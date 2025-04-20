## Classifier-guided CLIP Distillation for Unsupervised Multi-label Classification (CVPR 2025)

__Official pytorch implementation of "Classifier-guided CLIP Distillation for Unsupervised Multi-label Classification"__
> Dongseob Kim<sup>\*,1 </sup>, Hyunjung Shim<sup>2 </sup> <br>
> <sup>1 </sup> Samsung Electronics, and <sup>2 </sup> Korea Advanced Institute of Science \& Technology <br>
>
> __Abstract__ _Multi-label classification is crucial for comprehensive image understanding, yet acquiring accurate annotations is challenging and costly. To address this, a recent study suggests exploiting unsupervised multi-label classification leveraging CLIP, a powerful vision-language model. Despite CLIP's proficiency, it suffers from view-dependent predictions and inherent bias, limiting its effectiveness. We propose a novel method that addresses these issues by leveraging multiple views near target objects, guided by Class Activation Mapping (CAM) of the classifier, and debiasing pseudo-labels derived from CLIP predictions. Our Classifier-guided CLIP Distillation (CCD) enables selecting multiple local views without extra labels and debiasing predictions to enhance classification performance. Experimental results validate our method's superiority over existing techniques across diverse datasets._

## Updates

20 April, 2025: Initial upload


## Installation
**Step 0.** Install PyTorch and Torchvision following [official instructions](https://pytorch.org/get-started/locally/), e.g.,

```shell
pip install torch torchvision
# FYI, we're using torch==1.13.1 and torchvision==0.14.1
# We used docker image pytorch:1.13.1-cuda11.6-cudnn8-devel
# or later version will be okay
```

**Step 1.** Install packages.
```shell
pip install ftfy regex tqdm munch 
# FYI, we're using mmcv-full==1.4.0 
```

**Step 2.** Download CLIP pretrained model.
```shell
mkdir pretrained
curl https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt --output pretrained/RN50x64.pt
```

**Step 3.** Install CARB.
```shell
git clone https://github.com/k0u-id/CCD.git
```

## Dataset Preparation & Pretrained Checkpoint
In our paper, we experiment with pascal VOC, MS COCO, and Nuswide.
please refer data/README.md

- Example directory hierarchy
  ```
  CCD
  |--- data
  |    |--- pascal
  |    |    |---Annotations
  |    |    |---ImageSets
  |    |    |---JPEGImages
  |    |    | ...
  |    |--- coco
  |    |    | ...
  |    |--- nuswide
  |    |    | ...
  |--- metadata
  |    |--- voc12
  |    |    |---formatted_train_images.npy
  |    |    |---formatted_train_labels.npy
  |    |    |---formatted_val_images.npy
  |    |    | ...
  |    |--- voc07
  |    |    | ...
  |    |--- coco
  |    |    | ...
  |    |--- nuswide
  |    |    | ...
  |--- results
  |    |--- output_dirs (excuted-date&time)
  |    | ...
  | ...
  ```

**metadata**
- [voc12](https://drive.google.com/drive/folders/1NSKtFqS7Y2x7o4j3NKSIfa9fBIP1D6Iv?usp=sharing)
- [voc07](https://drive.google.com/drive/folders/1gbTkK4pEN83UbsNCBIXap1MBdb3XwgHA?usp=sharing)
- [coco](https://drive.google.com/drive/folders/11nyPoG9SaYZBQZzeTE2jfMhmCCafdlkD?usp=drive_link)
- [nuswide](https://drive.google.com/drive/folders/1dveiUQNT9dlvy0QSBkncPzfgqC74a-uo?usp=sharing)

**Pretrained Checkpoint**
- [voc12](https://drive.google.com/file/d/1JV2YcVqR38K5WU4D-g_frNOkDZ6SpGRH/view?usp=sharing)
- [voc07](https://drive.google.com/file/d/1HVoCA59JAYGjGFRD11NyAJ4T7lFB3Z_i/view?usp=sharing)
- [coco](https://drive.google.com/file/d/1N20oFJJ5QRwY5A09NxkVpIzujRT8z4ED/view?usp=sharing)
- [nuswide](https://drive.google.com/file/d/1kKDO5WojtzwFUjen9C9rJJWrOElfPSVz/view?usp=sharing)

## training CCD
CCD trains multi-label classification model with label updating.
```shell
# Please see this file for the detail of execution.
# You can change detailed configuration by changing config files (e.g., CARB/configs/carb/cityscapes_carb_dual.py)
bash train_gpu#0.sh 
```

## Inference CCD
```shell
# uncomment run_test in main_multi.py
# change line 59 in test.py
# we will refine inference code soon
```

## Acknoledgement
This code is highly borrowed from [Multi-Label Learning from Single Positive Labels](https://github.com/elijahcole/single-positive-multi-label), [BridgeGapExplanationPAMC](https://github.com/youngwk/BridgeGapExplanationPAMC). 

## Citation
If you use CCD or this code base in your work, please cite
```
@article{kim2025classifier,
  title={Classifier-guided CLIP Distillation for Unsupervised Multi-label Classification},
  author={Kim, Dongseob and Shim, Hyunjung},
  journal={arXiv preprint arXiv:2503.16873},
  year={2025}
}
```