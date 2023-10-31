# Leaded Gradient Method (LGM)

[[中文版]](README_zh.md)

<p float="left">
    <img src="img/figure1.png" width="400"/>
</p>

This repository contains the PyTorch implementation for paper **Dynamics-aware Adversarial Attack of Adaptive Neural Networks**

**Authors:** An Tao, Yueqi Duan, Yingqi Wang, Jiwen Lu, Jie Zhou

[[arxiv]](https://arxiv.org/abs/2210.08159) 

In this paper, we investigate the dynamics-aware adversarial attack problem of adaptive neural networks. Most existing adversarial attack algorithms are designed under a basic assumption -- the network architecture is fixed throughout the attack process. However, this assumption does not hold for many recently proposed adaptive neural networks, which adaptively deactivate unnecessary execution units based on inputs to improve computational efficiency. It results in a serious issue of lagged gradient, making the learned attack at the current step ineffective due to the architecture change afterward. To address this issue, we propose a Leaded Gradient Method (LGM) and show the significant effects of the lagged gradient. 

If you find our work useful in your research, please consider citing our paper.

```
@article{tao2022dynamicsaware,
  title={Dynamics-aware Adversarial Attack of Adaptive Neural Networks},
  author={Tao, An and Duan, Yueqi and Yingqi, Wang and Lu, Jiwen and Zhou, Jie},
  journal={https://arxiv.org/abs/2210.08159},
  year={2022}
}
```

&nbsp;

**Updates:** 

- [2023/10/31] Add attack code for 2D image classification.
- [2023/06/20] Add attack code for 3D point cloud outdoor scene semantic segmentation.
- [2022/10/17] The journal preprint version of this paper is available on arXiv.
- [2022/05/20] Add attack code for 3D point cloud indoor scene semantic segmentation.

&nbsp;

## Contents

- [2D Image Classification](2D/)
- [3D Point Cloud Indoor Scene Segmentation](3D/indoor_scene/)
- [3D Point Cloud Outdoor Scene Segmentation](3D/outdoor_scene/)
