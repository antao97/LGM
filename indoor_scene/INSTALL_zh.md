# 安装示例

[[English]](INSTALL.md)

在此列出在我们CentOS 7.8.2003系统中使用的包：

- CUDA 10.1 (cudatoolkit 10.1.243)
- python 3.7.10
- GCC 7.5
- pytorch 1.6.0
- torchvision 0.7.0
- numpy 1.20.2
- mkl-include 2021.2.0
- minkowskiengine 0.4.3
- plyfile 0.7.4
- torch_scatter 2.0.7
- open3d 0.9.0

对我们的系统安装这些包的步骤如下：

1. 安装GCC 7.5并激活，我们推荐一个[中文教程](https://blog.csdn.net/sinat_18697811/article/details/127448506)。激活后，使用`gcc -v`检查你的GCC版本是否正确。

2. 从[这里](https://developer.nvidia.com/cuda-10.1-download-archive-update2?target_os=Linux&target_arch=x86_64&target_distro=CentOS&target_version=7&target_type=runfilelocal)下载CUDA 10.1并安装，如果你希望不使用`sudo`权限来安装，我们推荐一个[中文教程](https://blog.csdn.net/qq_35498453/article/details/110532839)。

3. 安装[Anaconda](https://www.anaconda.com/)。新建一个新的conda环境，在新的环境中安装所需要的包，具体命令如下。

``` 
conda create -n py3-mink-43 python=3.7
conda activate py3-mink-43
conda install numpy==1.20.2
conda install mkl-include==2021.2.0
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
``` 

4. 从[这里](https://github.com/NVIDIA/MinkowskiEngine/archive/refs/tags/v0.4.3.zip)下载0.4.3版本的Minkowski Engine的`.zip` 安装压缩包并解压，然后使用如下命令安装。

``` 
cd MinkowskiEngine-0.4.3
python setup.py install --force_cuda --cuda_home <cuda path>
``` 

5. 安装一些额外需要的包，命令如下。

```
pip install plyfile==0.7.4
pip install torch_scatter==2.0.7
pip install open3d==0.9.0
```

请确保你使用了0.9.0版本的open3d，否则你会在运行程序时遇到问题。