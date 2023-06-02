''' Configurations for the ScanNet dataset

Dynamics-aware Adversarial Attack of 3D Sparse Convolution Network

Author: An Tao, Pengliang Ji
Email: ta19@mails.tsinghua.edu.cn, jpl1723@buaa.edu.cn
Date: 2022/1/13

'''

import argparse

parser = argparse.ArgumentParser()

### General Settings
parser.add_argument('--exp_name', type=str, default=None, 
                    help='Name of the experiment.')
parser.add_argument('--data_path', type=str,
                    default='/data3/antao/Documents/Datasets/'
                            'ScanNet_processed/train',
                    help='Data path to the dataset.')
parser.add_argument('--save_preds', action='store_true',
                    help='Whether to save point cloud coordinates. Default is False.')
parser.add_argument('--save_probs', action='store_true',
                    help='Whether to save the class prediction results. Default is False.')
parser.add_argument('--visual', action='store_true',
                    help='Whether to save the visualization results in `.ply` files. Default is False.')

### Model Settings
parser.add_argument('--voxel_size', type=float, default=0.02,
                    help='Voxel size for the model.')
parser.add_argument('--bn_momentum', type=float, default=0.05,
                    help='Batch normalization momentum.')
parser.add_argument('--conv1_kernel_size', type=int, default=5,
                    help='Convolution kernel size in the first convolution operation.')
parser.add_argument('--weights', type=str,
                    default='weights/MinkUNet34C-train-conv1-5.pth',
                    help='Path of weights to load for the model.')
                    
### Attack Settings
parser.add_argument('--budget', type=float, default=0.005,
                    help='Attack budget in L_inf (maximum perturbation).')
parser.add_argument('--dynamics_aware', type=bool, default=True,
                    help='Whether the attack is dynamics-aware. Default is True.')
parser.add_argument('--save_coords', action='store_true',
                    help='Whether to save attacked point cloud coordinates. Default is False.')
parser.add_argument('--resume_path', type=str, default=None,
                    help='Resume the attack with an experiment path. The format of the path is \
                    `outputs/scannet/budget_<your budget>/<your exp name>`. You need to make sure that \
                    you have used --save_coords in the resumed attack.')

### Attack Parameters
# If the attack budget lies in [0.005, 0.01, 0.02, 0.05], our script default loads our fine-tuned attack parameters. 
# You can change them on your own with the following arguments.
parser.add_argument('--default_para', type=bool, default=True,
                    help='Whether to use default attack parameters for budget that lies in [0.005, 0.01, 0.02, 0.05]. Default is True.')
parser.add_argument('--iter_num', type=int, default=None,
                    help='The iteration numer for attack.')
parser.add_argument('--step', type=float, default=None,
                    help='The step size for each attack step.')
parser.add_argument('--lamda_input', type=float, default=None,
                    help='This parameter controls the slop of the sigmoid-like function in input voxelization.')
parser.add_argument('--lamda_conv', type=float, default=None,
                    help='This parameter controls the slop of the sigmoid-like function in the occupancy value in sparse convolution.')
parser.add_argument('--lamda_output', type=float, default=None,
                    help='This parameter controls the slop of the sigmoid-like function in output devoxelization.')

### Evaluation Settings
parser.add_argument('--split', type=str, default='val', choices=['train', 'val'],
                    help='Which split to evaluate.')
parser.add_argument('--attacked_coords', type=str, default=None,
                    help='Evaluate the model performance with attacked point cloud coordinates.\
                    The format of path is `outputs/scannet/budget_<your budget>/<your exp name>/coord`.')


### Default Parameter List
parameters = {
    0.005: {
        True: {
            'iter_num': 30, 
            'step': 0.0025, 
            'lamda_input': 10, 
            'lamda_conv': 35, 
            'lamda_output': 10
        },
        False: {
            'iter_num': 30, 
            'step': 0.005, 
            'lamda_input': 20, 
            'lamda_conv': None, 
            'lamda_output': 20
        }
    },
    0.01: {
        True: {
            'iter_num': 50, 
            'step': 0.01, 
            'lamda_input': 10, 
            'lamda_conv': 10, 
            'lamda_output': 10
        },
        False: {
            'iter_num': 30, 
            'step': 0.01, 
            'lamda_input': 20, 
            'lamda_conv': None, 
            'lamda_output': 20
        }
    },
    0.02: {
        True: {
            'iter_num': 50, 
            'step': 0.01, 
            'lamda_input': 10, 
            'lamda_conv': 10, 
            'lamda_output': 10
        },
        False: {
            'iter_num': 30, 
            'step': 0.01, 
            'lamda_input': 20, 
            'lamda_conv': None, 
            'lamda_output': 20
        }
    },
    0.05: {
        True: {
            'iter_num': 50, 
            'step': 0.01, 
            'lamda_input': 10, 
            'lamda_conv': 35, 
            'lamda_output': 10
        },
        False: {
            'iter_num': 30, 
            'step': 0.05, 
            'lamda_input': 20, 
            'lamda_conv': None, 
            'lamda_output': 20
        }
    }
}
