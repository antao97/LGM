# Attack of 3D Point Cloud Indoor Scene Segmentation

[[中文版]](README_zh.md)

<p float="left">
    <img src="../image/indoor_scene.jpg" width="800"/>
</p>

This folder contains codes for the [ScanNet](http://kaldir.vc.in.tum.de/scannet_benchmark/documentation) and [Stanford (S3DIS)](http://buildingparser.stanford.edu/dataset.html) datasets. The codes are built from [SpatioTemporalSegmentation](https://github.com/chrischoy/SpatioTemporalSegmentation).

&nbsp;

## Requirements
Please follow the instructions in [SpatioTemporalSegmentation](https://github.com/chrischoy/SpatioTemporalSegmentation) to install required packages.

We recommend using [Anaconda](https://www.anaconda.com/) to install Minkowski Engine. Detailed steps can be found in [Minkowski Engine 0.4.3](https://github.com/NVIDIA/MinkowskiEngine/tree/v0.4.3#anaconda). You need to make sure that you download the 0.4.3 version, not the latest version.

Here we provide an installation example in this [link](INSTALL.md).

&nbsp;

## ScanNet Dataset

### Dataset

1. Download the ScanNet dataset from the [official website](http://kaldir.vc.in.tum.de/scannet_benchmark/documentation). You need to sign the terms of use.
2. Preprocess all the raw point clouds with the following command after you set the path correctly. You need to change the paths for `SCANNET_RAW_PATH` and `SCANNET_OUT_PATH` in `lib/datasets/preprocessing/scannet.py`.

``` 
python -m lib.datasets.preprocessing.scannet
``` 

### Model

In our work, we adopt the Mink16UNet34C	model with 2cm voxel size and 5 conv1 kernels in the [model zoo](https://github.com/chrischoy/SpatioTemporalSegmentation#model-zoo). The trained model `MinkUNet34C-train-conv1-5.pth` can be downloaded [here](https://node1.chrischoy.org/data/publications/minknet/MinkUNet34C-train-conv1-5.pth). Please place it under `weights/` folder.

### Start Attack

Use the following command to attack with an assigned budget `<budget>`. The data path `<data path>` is `SCANNET_OUT_PATH/train`.
```
python adv_scannet.py --data_path <data path> --budget <budget>
``` 

You need to use `CUDA_VISIBLE_DEVICES` to specify the GPU cards in the experiment, otherwise the script will use all GPU cards. For example, add `CUDA_VISIBLE_DEVICES=0,1,2,3` before the above command to use 4 GPU cards with card index `0,1,2,3`, or add `os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"` in your script.

You can add additional arguments at the end of the above command to control the attack process.

- `--exp_name <name>` Assign an experiment name. Default is `Logs_<date>_<time>`.
- `--dynamics_aware` Whether the attack is dynamics-aware. Default is `True`.
- `--save_coords` Whether to save attacked point cloud coordinates. Default is `False`.
- `--save_preds` Whether to save the class prediction results for the attacked network. Default is `False`.
- `--save_probs` Whether to save the probability values of classes for the attacked network. Default is `False`.
- `--visual` Whether to save the visualization results in `.ply` files. Default is `False`.

### Resume Attack

We recommend using `--save_coords` during the attack process. With the saved attacked point cloud coordinates, you can resume the attack using `--resume_path <resume path>`. Also, you can use the saved attacked point cloud coordinates to run the evaluation script `eval_scannet.py` in the next section.

- `--resume_path <resume path>` Resume the attack with an experiment path. The format of the path is `outputs/scannet/budget_<your budget>/<your exp name>`. You need to make sure that you have used `--save_coords` in the resumed attack.

### Change Attack Parameters

If the attack budget lies in [0.005, 0.01, 0.02, 0.05], our script default loads our fine-tuned attack parameters. You can change them on your own with the following arguments.

- `--default_para` Whether to use default attack parameters for budget that lies in [0.005, 0.01, 0.02, 0.05]. Default is `True`.
- `--iter_num <num>` The iteration numer for attack.
- `--step <size>` The step size for each attack step.
- `--lamda_input <value>` This parameter controls the slop of the sigmoid-like function in input voxelization.
- `--lamda_conv <value>` This parameter controls the slop of the sigmoid-like function in the occupancy value in sparse convolution.
- `--lamda_output <value>` This parameter controls the slop of the sigmoid-like function in output devoxelization.

### Evaluation

Use the following command to evaluate the model performance with unattacked point cloud coordinates.

```
python eval_scannet.py --data_path <data path>
```

Our script can reproduce 72.22% mIoU for `MinkUNet34C-train-conv1-5.pth` on the validation set.

After the attack process. You can use the following command to evaluate the model performance with attacked point cloud coordinates in path `<coord path>`. The format of `<coord path>` is `outputs/scannet/budget_<your budget>/<your exp name>/coord`

```
python eval_scannet.py --data_path=<data path> --attacked_coords=<coord path>
```

Also, you can add additional arguments at the end of the above two commands to control the evaluation process.

- `--exp_name <name>` Assign an experiment name. Default is `Logs_<date>_<time>`.
- `--save_preds` Whether to save the class prediction results. Default is `False`.
- `--save_probs` Whether to save the probability values of classes. Default is `False`.
- `--visual` Whether to save the visualization results in `.ply` files. Default is `False`.

### Performance

| Method | Budget = 0.005 m | Budget = 0.01 m | Budget = 0.02 m | Budget = 0.05 m | 
| :---: | :---: | :---: | :---: | :---: | 
| FGM | 60.44 | 55.51 | 38.65 | 8.70 | 
| LGM | **25.79** | **11.51** | **5.76** | **3.83** | 

Because we have modified some mistakes in our codes, the attack performance with the codes in this repo can be slightly better than the reported performance above in our paper. 

&nbsp;

## Stanford 3D Dataset (S3DIS)

### Dataset

1. Download the Stanford 3D (S3DIS) dataset from the [official website](http://buildingparser.stanford.edu/dataset.html). You need to sign the terms of use.
2. Preprocess all the raw point clouds with the following command after you set the path correctly. You need to change the paths for `STANFORD_3D_IN_PATH` and `STANFORD_3D_OUT_PATH` in `lib/datasets/preprocessing/stanford.py`.

``` 
python -m lib.datasets.preprocessing.stanford
``` 

### Model

In our work, we adopt the Mink16UNet34 model with 5cm voxel size and 5 conv1 kernels in the [model zoo](https://github.com/chrischoy/SpatioTemporalSegmentation#model-zoo). The trained model `Mink16UNet34-stanford-conv1-5.pth` can be downloaded [here](https://node1.chrischoy.org/data/publications/minknet/Mink16UNet34-stanford-conv1-5.pth). Please place it under `weights/` folder.

### Start Attack

Use the following command to attack with an assigned budget `<budget>`. The data path `<data path>` is `STANFORD_3D_OUT_PATH`.

```
python adv_stanford.py --data_path <data path> --budget <budget> 
``` 

You need to use `CUDA_VISIBLE_DEVICES` to specify the GPU cards in the experiment, otherwise the script will use all GPU cards. For example, add `CUDA_VISIBLE_DEVICES=0,1,2,3` before the above command to use 4 GPU cards with card index `0,1,2,3`, or add `os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"` in your script.

You can add additional arguments at the end of the above command to control the attack process.

- `--exp_name <name>` Assign an experiment name. Default is `Logs_<date>_<time>`.
- `--dynamics_aware` Whether the attack is dynamics-aware. Default is `True`.
- `--save_coords` Whether to save attacked point cloud coordinates. Default is `False`.
- `--save_preds` Whether to save the class prediction results for the attacked network. Default is `False`.
- `--save_probs` Whether to save the probability values of classes for the attacked network. Default is `False`.
- `--visual` Whether to save the visualization results in `.ply` files. Default is `False`.

### Resume Attack

We recommend using `--save_coords` during the attack process. With the saved attacked point cloud coordinates, you can resume the attack using `--resume_path <resume path>`. Also, you can use the saved attacked point cloud coordinates to run the evaluation script `eval_stanford.py` in the next section.

- `--resume_path <resume path>` Resume the attack with an experiment path. The format of the path is `outputs/stanford/budget_<your budget>/<your exp name>`. You need to make sure that you have used `--save_coords` in the resumed attack.

### Change Attack Parameters

If the attack budget lies in [0.005, 0.01, 0.02, 0.05], our script default loads our fine-tuned attack parameters. You can change them on your own with the following arguments.

- `--default_para` Whether to use default attack parameters for budget that lies in [0.005, 0.01, 0.02, 0.05]. Default is `True`.
- `--iter_num <num>` The iteration numer for attack.
- `--step <size>` The step size for each attack step.
- `--lamda_input <value>` This parameter controls the slop of the sigmoid-like function in input voxelization.
- `--lamda_conv <value>` This parameter controls the slop of the sigmoid-like function in the occupancy value in sparse convolution.
- `--lamda_output <value>` This parameter controls the slop of the sigmoid-like function in output devoxelization.
- `--lamda_floor <value>` This parameter controls the slop of the sigmoid-like function to mimic the floor function.

### Evaluation

Use the following command to evaluate the model performance with unattacked point cloud coordinates.

```
python eval_stanford.py --data_path <data path>
```

Our script can reproduce 65.47% mIoU for `Mink16UNet34-stanford-conv1-5.pth` on area 5.

After the attack process. You can use the following command to evaluate the model performance with attacked point cloud coordinates in path `<coord path>`. The format of `<coord path>` is `outputs/stanford/budget_<your budget>/<your exp name>/coord`

```
python eval_stanford.py --data_path=<data path> --attacked_coords=<coord path>
```

Also, you can add additional arguments at the end of the above two commands to control the evaluation process.

- `--exp_name <name>` Assign an experiment name. Default is `Logs_<date>_<time>`.
- `--area <id>` The area to use in the evaluation. Default is `5`.
- `--save_preds` Whether to save the class prediction results. Default is `False`.
- `--save_probs` Whether to save the probability values of classes. Default is `False`.
- `--visual` Whether to save the visualization results in `.ply` files. Default is `False`.

### Performance

| Method | Budget = 0.005 m | Budget = 0.01 m | Budget = 0.02 m | Budget = 0.05 m | 
| :---: | :---: | :---: | :---: | :---: | 
| FGM | 57.53 | 52.35 | 45.24 | 21.21 | 
| LGM | **48.20** | **39.65** | **30.93** | **7.45** | 

Because we have modified some mistakes in our codes, the attack performance with the codes in this repo can be slightly better than the reported performance above in our paper. 
