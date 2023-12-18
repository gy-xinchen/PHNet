

# PAHNet: A pulmonary arterial hypertension detection network based on cine cardiac magnetic resonance images using a hybrid strategy of adaptive triplet and binary cross-entropy losses

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

**Description**

Pulmonary arterial hypertension (PAH) detection by cine cardiac magnetic resonance(CMR) images is one of the most important methods for noninvasive screening potential patients.  Machine learning(ML) can identify cine CMR image features to alert radiologist to the presence of PAH.  However, detecting mild PAH still poses a significant challenge. In this paper, we propose a convolutional neural network(CNN) based detection network with a hybrid strategy of adaptive triplet and binary cross-entropy losses(HSATBCL) to detect PAH from cine CMR images called PAHNet. Unlike previous research, our approach involves direct extraction of deep features from cine CMR images for PAH diagnosis. Meanwhile, we design HSATBCL to optimize model detection performance by building a triplet contrastive learning idea to learning mild PAH deep features. In internal validating environment, experiments show that the PAHNet could achieve an average Area Under Curve(AUC) value of 0.964, an accuracy of 0.912, and an F1-score of 0.884 in comparison with two state-of-the-art ML models and four typical deep learning CNN models. PAHNet also achieve the average AUC value of 0.828 by performing transfer learning on a public PAH dataset (Shef179-PAH dataset) using model weights. Experimental results indicate that the proposed PAHNet achieves superior performance to state-of-the-art models in PAH detection. Thus, it has great potential to reduce the misdiagnosis of PAH using cine CMR images in clinical practice.

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
