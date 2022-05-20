"""

Dynamics-aware Adversarial Attack of 3D Sparse Convolution Network

@Author: 
    An Tao,
    Pengliang Ji

@Contact: 
    ta19@mails.tsinghua.edu.cn, 
    jpl1723@buaa.edu.cn
    
@Time: 
    2022/1/23 9:32 PM

"""

import argparse

parser = argparse.ArgumentParser()

# General Settings
parser.add_argument('--exp_name', type=str, default=None, 
                    help='Name of the experiment')
parser.add_argument('--data_path', type=str,
                    default='/data3/antao/Documents/Datasets/'
                            'S3DIS_processed/')
parser.add_argument('--save_preds', default=False, action='store_true')
parser.add_argument('--save_probs', default=False, action='store_true')
parser.add_argument('--visual', default=False, action='store_true')

# Model Settings
parser.add_argument('--voxel_size', type=float, default=0.05)
parser.add_argument('--bn_momentum', type=float, default=0.05)
parser.add_argument('--conv1_kernel_size', type=int, default=5)
parser.add_argument('--weights', type=str,
                    default='weights/Mink16UNet34-stanford-conv1-5.pth')
                    
# Attack Settings
parser.add_argument('--budget', type=float, default=0.005)
parser.add_argument('--dynamics_aware', type=bool, default=True)
parser.add_argument('--save_coords', default=False, action='store_true')
parser.add_argument('--resume_path', type=str, default=None)

# Attack Parameters
parser.add_argument('--default_para', type=bool, default=True)
parser.add_argument('--iter_num', type=int, default=None)
parser.add_argument('--step', type=float, default=None)
parser.add_argument('--lamda_input', type=float, default=None)
parser.add_argument('--lamda_conv', type=float, default=None)
parser.add_argument('--lamda_output', type=float, default=None)
parser.add_argument('--lamda_floor', type=float, default=None)

# Evaluation Settings
parser.add_argument('--area', type=str, default='5', choices=['1', '2', '3', '4', '5', '6'])
parser.add_argument('--attacked_coords', type=str, default=None)


# Default Parameter List
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
