# 三维点云室外场景分割的攻击

[[English]](README.md)

<p float="left">
    <img src="../../../img/outdoor.jpg" width="800"/>
</p>

本文件夹包括了[SemanticKITTI](http://www.semantic-kitti.org/)数据集的代码，代码依据[Cylinder3D](https://github.com/xinge008/Cylinder3D)代码库建立。

&nbsp;

### 运行需求
请依据[Cylinder3D](https://github.com/xinge008/Cylinder3D)代码库中指示安装所需要的包。在本文件夹，代码使用`spconv 2.3.6`建立，使用了`cuda 11.6`。


### 数据集

1. 从[官方网站](http://www.semantic-kitti.org/dataset.html#download)下载SemanticKITTI数据集。
2. 在`config/semantickitti.yaml`中设置你的数据集路径

### 模型

在本工作中，我们使用了Cylinder3D官方提供的[预训练模型](https://github.com/xinge008/Cylinder3D#pretrained-models)，记得在`config/semantickitti.yaml`中更改你的模型路径。

### 开始攻击

在给定最大攻击幅度`<budget>`下，使用如下命令使用FGM攻击。

```
python fgm.py --budget <budget>
``` 

在给定最大攻击幅度`<budget>`下，使用如下命令使用LGM攻击。

```
python lgm.py --budget <budget>
``` 

你需要使用`CUDA_VISIBLE_DEVICES`去指定实验中具体使用哪些GPU卡。

你可以通过在上述命令的末尾增加下面这些额外的指令来控制攻击的具体过程。

- `--exp_name <name>` 指定一个实验名称，如果不指定则实验名称默认是`Logs_<日期>_<时间>`。
- `--save_coords` 是否保存攻击后的点云坐标，默认是`False`。
- `--save_preds` 是否保存攻击后模型的预测结果，默认是`False`。
- `--save_probs` 是否保存攻击后模型的各类别预测分数，默认是`False`。

### 恢复攻击

我们建议您在攻击时使用`--save_coords`命令，这样攻击时可以保存攻击过后的点云坐标。当程序意外终止时，通过这些已保存的点云坐标您可以在攻击命令中添加命令`--resume_path <resume path>`恢复上次攻击。

- `--resume_path <resume path>` 恢复某次攻击实验，路径的格式是`outputs/budget_<your budget>/<your exp name>`。你需要确保在需要恢复的攻击实验中，之前已经使用了`--save_coords`命令。

### 改变攻击参数

如果攻击的最大攻击幅度是列表[0.005, 0.01, 0.02, 0.05]中的一个，脚本将自动加载我们调好的攻击参数，您可以根据您的需要通过下面的指令修改攻击参数。

- `--default_para` 当攻击的最大攻击幅度是列表[0.005, 0.01, 0.02, 0.05]中的一个，是否使用默认攻击参数，默认是`True`。
- `--iter_num <num>` 攻击迭代数量。
- `--step <size>` 攻击步长。
- `--lamda_input <value>` 本参数控制在模型输入体素化（voxelization）处S形函数（sigmoid-like function）的斜度。
- `--lamda_conv <value>` 本参数控制在稀疏卷积的占用值（occupancy value）处S形函数（sigmoid-like function）的斜度。

### 其他

使用如下命令使用未攻击的点云坐标来测试模型性能，输出结果反映了攻击前模型的性能。

```
python test.py
```

我们的脚本能够复现出mIoU 66.91%性能。

### 性能

| 方法 | 最大攻击幅度 = 0.005 m | 最大攻击幅度 = 0.01 m | 最大攻击幅度 = 0.02 m | 最大攻击幅度 = 0.05 m | 
| :---: | :---: | :---: | :---: | :---: | 
| FGM | 34.43 | 29.77 | 16.02 | 9.17 | 
| LGM | 33.63 | 29.13 | 15.17 | 8.50 | 
