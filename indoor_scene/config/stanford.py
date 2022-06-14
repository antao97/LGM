''' Configurations for the Stanford dataset (S3DIS)

Dynamics-aware Adversarial Attack of 3D Sparse Convolution Network

Author: An Tao, Pengliang Ji
Email: ta19@mails.tsinghua.edu.cn, jpl1723@buaa.edu.cn
Date: 2022/1/13

'''

import argparse

parser = argparse.ArgumentParser()

### General Settings
parser.add_argument('--exp_name', type=str, default=None, 
                    help='Name of the experiment')
parser.add_argument('--data_path', type=str,
                    default='/data3/antao/Documents/Datasets/'
                            'S3DIS_processed/',
                    help='Data path to the dataset.')
parser.add_argument('--save_preds', default=False, action='store_true',
                    help='Whether to save point cloud coordinates. Default is False.')
parser.add_argument('--save_probs', default=False, action='store_true',
                    help='Whether to save the class prediction results. Default is False.')
parser.add_argument('--visual', default=False, action='store_true',
                    help='Whether to save the visualization results in `.ply` files. Default is False.')

### Model Settings
parser.add_argument('--voxel_size', type=float, default=0.05,
                    help='Voxel size for the model.')
parser.add_argument('--bn_momentum', type=float, default=0.05,
                    help='Batch normalization momentum.')
parser.add_argument('--conv1_kernel_size', type=int, default=5,
                    help='Convolution kernel size in the first convolution operation.')
parser.add_argument('--weights', type=str,
                    default='weights/Mink16UNet34-stanford-conv1-5.pth',
                    help='Path of weights to load for the model.')          
                    
### Attack Settings
parser.add_argument('--budget', type=float, default=0.005,
                    help='Attack budget in L_inf (maximum perturbation).')
parser.add_argument('--dynamics_aware', type=bool, default=True,
                    help='Whether the attack is dynamics-aware. Default is True.')
parser.add_argument('--save_coords', default=False, action='store_true',
                    help='Whether to save attacked point cloud coordinates. Default is False.')
parser.add_argument('--resume_path', type=str, default=None,
                    help='Resume the attack with an experiment path. The format of the path is \
                    `outputs/stanford/budget_<your budget>/<your exp name>`. You need to make sure that \
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
parser.add_argument('--lamda_floor', type=float, default=None,
                    help='This parameter controls the slop of the sigmoid-like function to mimic the floor function.')

### Evaluation Settings
parser.add_argument('--area', type=str, default='5', choices=['1', '2', '3', '4', '5', '6'],
                    help='Load which area in the dataset.')
parser.add_argument('--attacked_coords', type=str, default=None,
                    help='Evaluate the model performance with attacked point cloud coordinates.\
                    The format of path is `outputs/stanford/budget_<your budget>/<your exp name>/coord.`')

### Default Parameter List
parameters = {
    0.005: {
        True: {
            'iter_num': 50, 
            'step': 0.005, 
            'lamda_input': 45, 
            'lamda_conv': 55, 
            'lamda_output': 45,
            'lamda_floor': 20
        },
        False: {
            'iter_num': 30, 
            'step': 0.005, 
            'lamda_input': 55, 
            'lamda_conv': None, 
            'lamda_output': 55,
            'lamda_floor': 20
        }
    },
    0.01: {
        True: {
            'iter_num': 50, 
            'step': 0.01, 
            'lamda_input': 45, 
            'lamda_conv': 55, 
            'lamda_output': 45,
            'lamda_floor': 20
        },
        False: {
            'iter_num': 30, 
            'step': 0.01, 
            'lamda_input': 55, 
            'lamda_conv': None, 
            'lamda_output': 55,
            'lamda_floor': 20
        }
    },
    0.02: {
        True: {
            'iter_num': 50, 
            'step': 0.01, 
            'lamda_input': 45, 
            'lamda_conv': 55, 
            'lamda_output': 45,
            'lamda_floor': 20
        },
        False: {
            'iter_num': 30, 
            'step': 0.02, 
            'lamda_input': 55, 
            'lamda_conv': None, 
            'lamda_output': 55,
            'lamda_floor': 20
        }
    },
    0.05: {
        True: {
            'iter_num': 50, 
            'step': 0.025, 
            'lamda_input': 45, 
            'lamda_conv': 55, 
            'lamda_output': 45,
            'lamda_floor': 20
        },
        False: {
            'iter_num': 30, 
            'step': 0.05, 
            'lamda_input': 55, 
            'lamda_conv': None, 
            'lamda_output': 55,
            'lamda_floor': 20
        }
    }
}
