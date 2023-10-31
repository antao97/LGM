''' Config file

Dynamics-aware Adversarial Attack of Adaptive Neural Networks

Author: An Tao, Yingqi Wang
Email: ta19@mails.tsinghua.edu.cn, yingqi-w19@mails.tsinghua.edu.cn
Date: 2023/10/31

'''

import argparse
import logging

def parse_args():
    parser = argparse.ArgumentParser(
        description='attack script')
    parser.add_argument('--model_path', default=None, type=str,
                        help='pretrained model path')
    parser.add_argument('--model', default='skipnet', type=str, choices=['skipnet', 'dynconv', 'resnet'],
                        help='attacked model') 
    parser.add_argument('--resnet', default=None, type=str, 
                        help='resnet model type') 
    parser.add_argument('--type', default='lgm', type=str, choices=['fgm', 'lgm'],
                        help='LGM or FGM attack') 
    parser.add_argument('--dataset', default='cifar', type=str, choices=['cifar', 'imagenet'],
                        help='dataset') 
    parser.add_argument('--loss', default='ce', type=str, choices=['ce', 'cw'],
                        help='loss criteration') 
    parser.add_argument('-t', default=False, action='store_true',
                        help='whether targeted attack.') 
    parser.add_argument('--mask_rate', default=1, type=float,
                        help='mask rate')                  
    parser.add_argument('--batch_size', default=16, type=int,
                        help='mini-batch size (default: 1)')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4 )')
    parser.add_argument('--lr', default=1, type=int,
                        help='attack lr size (default: 1)')
    parser.add_argument('--eps', default=8, type=int,
                        help='attack eps (default: 8)')
    parser.add_argument('--n_iter', default=1, type=int,
                        help='number of total iterations '
                             '(previous default: 1)')
    parser.add_argument('--lamb', default=None, type=float,
                        help='lambda in sigmoid function of LGM (default: 10)')
    parser.add_argument('--diff_type', default='gg', type=str, choices=['gg', 'fg'],
                        help='differential type in LGM (default: gg)')
    parser.add_argument('--momentum', default=1, type=float,
                        help='momentum in gradient (default: 1)') 
    parser.add_argument('-m', default=False, action='store_true',
                        help='whether to use momentum in gradient') 
    parser.add_argument('-nes', default=False, action='store_true',
                        help='whether to use Nesterov Accelerated Gradient.')
    parser.add_argument('-pgd', default=False, action='store_true',
                        help='whether random initial point. (PGD method)')
    parser.add_argument('-adam', default=False, action='store_true',
                        help='whether use Adam optimizer.') 
    parser.add_argument('-auto_lr', default=False, action='store_true',
                        help='whether use AutoPGD.')
    parser.add_argument('--exp', default=None, type=str,
                        help='experiment name') 
    parser.add_argument('--save', default=False, action='store_true')
    
    args = parser.parse_args()
    return args  

cifar_resnet_dict = {
    '20': 'resnet20-12fca82f.th',
    '32': 'resnet32-d509ac18.th',
    '44': 'resnet44-014dd654.th',
    '56': 'resnet56-4bfd9763.th',
    '110': 'resnet110-1d1ed7c2.th',
    '1202': 'resnet1202-f3b1deed.th'
    }
