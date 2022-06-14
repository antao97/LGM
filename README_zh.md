# 超前梯度法（LGM）

[[English]](README.md)

<p float="left">
    <img src="image/figure1.jpg" width="700"/>
</p>

本代码库包含了文章**Dynamics-aware Adversarial Attack of 3D Sparse Convolution Network**的PyTorch实现代码。

**作者：** 陶安，段岳圻，王鹤，吴紫屹，纪鹏亮，孙浩文，周杰，鲁继文

[[论文]](https://arxiv.org/abs/2112.09428)

在论文中，我们研究了深度神经网络中的动态感知对抗攻击问题（Dynamics-aware Adversarial Attack）。大多数现有的对抗性攻击算法都是在一个基本假设下设计的——网络架构在整个攻击过程中都是固定的。然而，这个假设不适用于许多最近提出的网络，例如三维稀疏卷积网络，其中包含依赖于输入的运算单元以提高总体计算效率。这种网络在攻击中存在严重的梯度滞后问题，即由于攻击前后的网络架构变化，使得当前步骤的学习攻击变得无效。为了解决这个问题，我们提出了一种超前梯度法（LGM），并在实验中展示了滞后梯度造成的显著影响。

如果您发现我们的工作对您的研究有帮助，您可以考虑引用我们的论文。

```
@article{tao2021dynamicsaware,
  title={Dynamics-aware Adversarial Attack of 3D Sparse Convolution Network},
  author={Tao, An and Duan, Yueqi and Wang, He and Wu, Ziyi and Ji, Pengliang and Sun, Haowen and Zhou, Jie and Lu, Jiwen},
  journal={arXiv preprint arXiv:2112.09428},
  year={2021}
}
```

&nbsp;

**更新：** 

- [2022/5/20] 增加三维点云室内场景分割的攻击代码。

&nbsp;

## 内容

- [三维点云室内场景分割](indoor_scene/)
