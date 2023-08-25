''' Configurations for the SemanticKITTI dataset

Dynamics-aware Adversarial Attack of Adaptive Neural Networks

Author: An Tao, Pengliang Ji
Email: ta19@mails.tsinghua.edu.cn, jpl1723@buaa.edu.cn
Date: 2023/6/15

'''

import argparse

attack_parser = argparse.ArgumentParser()

### General Settings
attack_parser.add_argument('--exp_name', type=str, default=None, 
                    help='Name of the experiment')
attack_parser.add_argument('--save_preds', action='store_true',
                    help='Whether to save point cloud coordinates. Default is False.')
attack_parser.add_argument('--save_probs', action='store_true',
                    help='Whether to save the class prediction results. Default is False.')

### Attack Settings
attack_parser.add_argument('--budget', type=float, default=0.01,
                    help='Attack budget in L_inf (maximum perturbation).')
attack_parser.add_argument('--save_coords', action='store_true',
                    help='Whether to save attacked point cloud coordinates. Default is False.')
attack_parser.add_argument('--resume_path', type=str, default=None,
                    help='Resume the attack with an experiment path. The format of the path is \
                    `outputs/scannet/budget_<your budget>/<your exp name>`. You need to make sure that \
                    you have used --save_coords in the resumed attack.')

### Attack Parameters
# If the attack budget lies in [0.005, 0.01, 0.02, 0.05], our script default loads our fine-tuned attack parameters. 
# You can change them on your own with the following arguments.
attack_parser.add_argument('--default_para', type=bool, default=True,
                    help='Whether to use default attack parameters for budget that lies in [0.005, 0.01, 0.02, 0.05]. Default is True.')
attack_parser.add_argument('--iter_num', type=int, default=None,
                    help='The iteration numer for attack.')
attack_parser.add_argument('--step', type=float, default=None,
                    help='The step size for each attack step.')
attack_parser.add_argument('--lamda_input', type=float, default=None,
                    help='This parameter controls the slop of the sigmoid-like function in input voxelization.')
attack_parser.add_argument('--lamda_conv', type=float, default=None,
                    help='This parameter controls the slop of the sigmoid-like function in the occupancy value in sparse convolution.')

### Default Parameter List
attack_parameters = {
    0.005: {
        'iter_num': 30, 
        'step': 0.0025, 
        'lamda_input': 10, 
        'lamda_conv': 250
    },
    0.01: {
        'iter_num': 30, 
        'step': 0.005, 
        'lamda_input': 10, 
        'lamda_conv': 200
    },
    0.02: {
        'iter_num': 30, 
        'step': 0.005, 
        'lamda_input': 10, 
        'lamda_conv': 200
    },
    0.05: {
        'iter_num': 30, 
        'step': 0.005, 
        'lamda_input': 10, 
        'lamda_conv': 200
    }
}
