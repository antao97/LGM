# An Installation Example

[[中文版]](INSTALL_zh.md)

Here we list the packages we use in our CentOS 7.8.2003 system:

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

We list the steps to install these packages for our system.

1. Install GCC 7.5 and activate it. We recommend a Chinese tutorial in this [link](https://blog.csdn.net/sinat_18697811/article/details/127448506). Check your GCC version with `gcc -v`.

2. Download CUDA 10.1 from this [link](https://developer.nvidia.com/cuda-10.1-download-archive-update2?target_os=Linux&target_arch=x86_64&target_distro=CentOS&target_version=7&target_type=runfilelocal) and install it. If you want to install it without `sudo`, we recommend a Chinese tutorial in this [link](https://blog.csdn.net/qq_35498453/article/details/110532839).

3. Install [Anaconda](https://www.anaconda.com/). Create a new conda environment and install the required packages.

``` 
conda create -n py3-mink-43 python=3.7
conda activate py3-mink-43
conda install numpy==1.20.2
conda install mkl-include==2021.2.0
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
``` 

4. Download the 0.4.3 version of Minkowski Engine in `.zip` format from this [link](https://github.com/NVIDIA/MinkowskiEngine/archive/refs/tags/v0.4.3.zip) and unzip it. Then use the following commands.

``` 
cd MinkowskiEngine-0.4.3
python setup.py install --force_cuda --cuda_home <cuda path>
``` 

5. Install some additional required packages.

```
pip install plyfile==0.7.4
pip install torch_scatter==2.0.7
pip install open3d==0.9.0
```

Make sure that you use the 0.9.0 version of open3d, otherwise you will encounter an error when you run the experiment.