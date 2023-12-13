# Attack of 2D Image Classification

[[中文版]](README_zh.md)

<p float="left">
    <img src="../img/imagenet.jpg" width="800"/>
</p>

This folder contains codes for the attacks of [SkipNet](https://github.com/ucbdrive/skipnet/) and [DynConv](https://github.com/thomasverelst/dynconv). 

&nbsp;

## Requirements

The codebase was developed and tested on CentOS 7.8, with GPU RTX_2080.

We provide an Anaconda environment with the dependencies, to install run

``` 
conda env create -f environment.yml
conda activate dyn_attack
``` 

&nbsp;

## Datasets

CIFAR-10 dataset will be downloaded automatically when running the code.

You need to follow this [link](https://github.com/facebookarchive/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset) to download ImageNet dataset. We only need the validation set. The downloaded dataset should be placed look like this:

```
data/val/
      |--n01440764/
      |--n01443537/
      |--...
```

&nbsp;

## Models

### SkipNet
Download trained checkpoints in [link](https://github.com/ucbdrive/skipnet/tree/master/cifar#demo) and [link](https://github.com/ucbdrive/skipnet/blob/master/imagenet/README.md). We use `resnet-110-rnn-cifar10.pth.tar` and `resnet-101-rnn-imagenet.pth.tar` in our attack. Place the checkpoints under `skipnet/checkpoints/`. 

### DynConv
Download trained checkpoints in [link](https://github.com/thomasverelst/dynconv/tree/master/classification#trained-models). We use ResNet32 with sparse05 for CIFAR-10 and ResNet101 with sparse05 for ImageNet. Place the checkpoints under `dynconv/`. 

### ResNet
Download trained checkpoints in [link](skipnet/cifar/pytorch_resnet_cifar10/README.md). We use ResNet32 and ResNet110 for CIFAR-10. Place the checkpoints under `skipnet/cifar/pytorch_resnet_cifar10/pretrained_models/`. The ResNet trained checkpoints for ImageNet will be downloaded automatically shen running the code.

&nbsp;

## Attack

### FGSM attack
- FGM

``` 
python attack.py --model <skipnet|dynconv|resnet> --dataset <cifar|imagenet> --type fgm --eps <epsilon> --lr <step size> --batch_size <batch size>
``` 

- LGM

``` 
python attack.py --model <skipnet|dynconv|resnet> --dataset <cifar|imagenet> --type lgm --eps <epsilon> --lr <step size> --batch_size <batch size> --lamb <lamb>
``` 

### BIM attack
- FGM

``` 
python attack.py --model <skipnet|dynconv|resnet> --dataset <cifar|imagenet> --type fgm --eps <epsilon> --batch_size <batch size> --n_iter <iter> --mask_rate <mask rate>
``` 

- LGM

``` 
python attack.py --model <skipnet|dynconv|resnet> --dataset <cifar|imagenet> --type lgm --eps <epsilon> --batch_size <batch size> --n_iter <iter> --mask_rate <mask rate> --lamb <lamb>
``` 

### PGD attack
- FGM

``` 
python attack.py --model <skipnet|dynconv|resnet> --dataset <cifar|imagenet> --type fgm --eps <epsilon> --batch_size <batch size> --n_iter <iter> --mask_rate <mask rate> -pgd
``` 

- LGM

``` 
python attack.py --model <skipnet|dynconv|resnet> --dataset <cifar|imagenet> --type lgm --eps <epsilon> --batch_size <batch size> --n_iter <iter> --mask_rate <mask rate> -pgd --lamb <lamb>
``` 

### CW attack
- FGM

``` 
python attack.py --model <skipnet|dynconv|resnet> --dataset <cifar|imagenet> --type fgm --eps <epsilon> --batch_size <batch size> --n_iter <iter> --mask_rate <mask rate> --loss cw -adam
``` 

- LGM

``` 
python attack.py --model <skipnet|dynconv|resnet> --dataset <cifar|imagenet> --type lgm --eps <epsilon> --batch_size <batch size> --n_iter <iter> --mask_rate <mask rate> --loss cw -adam --lamb <lamb>
``` 

### APGD attack
- FGM

``` 
python attack.py --model <skipnet|dynconv|resnet> --dataset <cifar|imagenet> --type fgm --eps <epsilon> --batch_size <batch size> --n_iter <iter> --mask_rate <mask rate> -auto_lr
``` 

- LGM

``` 
python attack.py --model <skipnet|dynconv|resnet> --dataset <cifar|imagenet> --type lgm --eps <epsilon> --batch_size <batch size> --n_iter <iter> --mask_rate <mask rate> -auto_lr --lamb <lamb>
``` 

Other arguments:

- `-t`: Targeted attack
- `--resnet <type>`: Specify resnet type when attacking ResNet, for example `--resnet 32`
- `--save`: Save attack results

&nbsp;

## Analysis

Run `analysis.py` to draw heatmap and T-SNE results.

&nbsp;

## Lambda of LGM
### Single Step Attack (FGSM)

- CIFAR-10

| Model | epsilon = 1 | epsilon = 2 | epsilon = 4 | epsilon = 8 | 
| :---: | :---: | :---: | :---: | :---: | 
| SkipNet | 0.5 | 0.3 | 0.2 | 0.2 | 
| DynConv | 0.5 | 0.2 | 0.1 | 0.0001 | 


- ImageNet

| Model | epsilon = 1 | epsilon = 2 | epsilon = 4 | epsilon = 8 | epsilon = 16 | 
| :---: | :---: | :---: | :---: | :---: | :---: | 
| SkipNet | 0.3 | 0.005 | 0.005 | 0.0025 | 0.0001 | 
| DynConv | 5 | 0.05 | 0.005 | 0.001 | 0.001 |

### Iterative Attack

- CIFAR-10 with valid pixel 5%

| Model | BIM | PGD | CW | APGD | BIM (Targeted) | PGD (Targeted) | CW (Targeted) | APGD (Targeted) | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| SkipNet | 5 | 5 | 35 | 5 | 5 | 5 | 30 | 5 | 
| DynConv | 10 | 10 | 45 | 10 | 10 | 10 | 45 | 10 | 
