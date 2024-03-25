

# PHNet: A pulmonary hypertension detection network based on cine cardiac magnetic resonance images using a hybrid strategy of adaptive triplet and binary cross-entropy losses

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

**Description**

Pulmonary hypertension (PH) detection by cine cardiac magnetic resonance(CMR) images is one of the most crucial methods for noninvasive screening of potential patients. Machine learning(ML) can identify cine CMR image features to alert radiologists to the presence of PH. However, detecting mild PH still poses a significant challenge. In this paper, we propose a nnU-Net-based segmentation network and a convolutional neural network(CNN) based detection network with a Hybrid Strategy of Adaptive Triplet and Binary Cross-entropy Losses(HSATBCL) to detect PH from cine CMR images called PHNet. Unlike previous research, our approach directly extracts deep features from cine CMR images for PH diagnosis.
We design HSATBCL to optimize model detection performance by building a triplet contrastive learning to pay attention to mild PH deep features. Experiments show that the PHNet could achieve an average area under the curve(AUC) value of 0.964, an accuracy of 0.912, and an F1-score of 0.884 in the internal validating environment. PHNet also achieves the average AUC value of 0.828 by performing transfer learning on a public PH dataset (Shef179-PH dataset) using model weights, indicating that the proposed PHNet performs superior to state-of-the-art cine CMR PH detection models. Thus, PHNet has great potential to reduce the misdiagnosis of PH using cine CMR images in clinical practice.


<p align="center">
  <a href="https://github.com/gy-xinchen/PAHNet/">
    <img src="imgs/Average_ROC.svg" alt="Logo" width="800" height="500">
  </a>

</p>

<p align="center">
  <a href="https://github.com/gy-xinchen/PAHNet/">
    <img src="imgs/transform.png" alt="Logo" width="800" height="350">
  </a>

</p>

<p align="center">
  <a href="https://github.com/gy-xinchen/PAHNet/">
    <img src="imgs/Visualization.svg" alt="Logo" width="1000" height="300">
  </a>

</p>

## PAHNet in PyTorch
We provide PyTorch implementations for PAHNet.
The code was inspired by [AdaTriplet](https://github.com/Oulu-IMEDS/AdaTriplet) and modified by [xinchen yuan](https://github.com/gy-xinchen).

**Note**: The current software works well with PyTorch 1.4.0+.

The 2D nnU-Net was a older nnU-Net created by [FabianIsensee](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1), it was used to biventricular segmentation with the 2017 Automated Cardiac Diagnosis Challenge(ACDC) dataset pre-trained weight from cine CMR images.

The following PAHNet was created by [xinchen yuan](https://github.com/gy-xinchen), it contains a biventricular segmentation module and DenseNet based Classification module.

<p align="center">
  <a href="https://github.com/gy-xinchen/PAHNet/">
    <img src="imgs/framework.svg" alt="Logo" width="1000" height="400">
  </a>

</p>

## Prerequisites
- Linux or Windows (Pycharm + Anaconda)
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
-You can install all the dependencies by
```bash
pip install -r requirements.txt
```
- Clone this repo:
```bash
git clone https://github.com/gy-xinchen/PHNet
cd PHNet
```
- For Anaconda users, you can use pip to install PyTorch and other libraries.

## [Datasets]
create a directory below and add your own datasets.

choosed slice cine CMR images have same size of 25x224x224.
```
Random_cine_CMR_Data：
|─train
│      patient000.nii.gz 
│      patient001.nii.gz
│      patient002.nii.gz
│      ...
│
├─internal_val
│      patient000.nii.gz 
│      patient001.nii.gz
│      patient002.nii.gz
│      ...
│
├─external_test
│      patient000.nii.gz 
│      patient001.nii.gz
│      patient002.nii.gz
│      ...
│
└─train_data.csv
└─val_data.csv
```

<!-- links -->
[your-project-path]:gy-xinchen/PAHNet
[contributors-shield]: https://img.shields.io/github/contributors/gy-xinchen/PAHNet.svg?style=flat-square
[contributors-url]: https://github.com/gy-xinchen/PAHNet/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/gy-xinchen/PAHNet.svg?style=flat-square
[forks-url]: https://github.com/gy-xinchen/PAHNet/network/members
[stars-shield]: https://img.shields.io/github/stars/gy-xinchen/PAHNet.svg?style=flat-square
[stars-url]: https://github.com/gy-xinchen/PAHNet/stargazers
[issues-shield]: https://img.shields.io/github/issues/gy-xinchen/PAHNet.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/gy-xinchen/PAHNet.svg
[license-shield]: https://img.shields.io/github/license/shaojintian/Best_README_template.svg?style=flat-square
[license-url]: https://github.com/gy-xinchen/PAHNet/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/gy-xinchen
