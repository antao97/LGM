''' Attack script

Dynamics-aware Adversarial Attack of Adaptive Neural Networks

Author: An Tao, Yingqi Wang
Email: ta19@mails.tsinghua.edu.cn, yingqi-w19@mails.tsinghua.edu.cn
Date: 2023/10/31

'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os
import sys
import time
import logging
import numpy as np
import torchvision
from utils import *
from config import *


def attack_process(args, targeted_class=None):
    if args.model == 'skipnet' and args.dataset == 'cifar':
        model, model_da = None, None
    else:
        model, model_da = load_model(args)

    logging.info(str(args))

    cudnn.benchmark = False

    if args.dataset == 'cifar':
        testset = torchvision.datasets.CIFAR10(root='/tmp/data',
                                               train=False,
                                               download=True,
                                               transform=transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=args.workers)
    elif args.dataset == 'imagenet':
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder('data/val', transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    
    torch.manual_seed(1)
    mask_idx = torch.randperm(args.image_size*args.image_size)[:int(args.mask_rate*args.image_size*args.image_size)]
    mask = torch.zeros(args.image_size*args.image_size)
    mask[mask_idx] = 1
    mask = mask.view(args.image_size, args.image_size)
    attacker = Attack(n_iter=args.n_iter, lr=args.lr, eps=args.eps, targeted=args.t, 
                                momentum=args.momentum, m=args.m, adam=args.adam, nes=args.nes, pgd=args.pgd, 
                                auto_lr=args.auto_lr, loss=args.loss, mask=mask, class_num=args.class_num, verbose=True)
    
    num_all = 0
    correct_all = 0
    correct_attack_all = 0
    total_layer_all = 0
    different_layer_all = 0
    for i, (input, target) in enumerate(test_loader):
        num, correct, correct_attack, different_layer, total_layer \
                = run_attack(args, model, model_da, i, input, target, len(test_loader), attacker, accuracy_detail, targeted_class)

        # torch.cuda.empty_cache()
        num_all += num
        correct_all += correct
        correct_attack_all += correct_attack
        total_layer_all += total_layer
        different_layer_all += different_layer
    if args.t:
        logging.info(args.log_head + '\tTargeted Class:{}\tAttack Success Rate:{:.2f}%\tDifferent Layer Rate:{:.2f}%'.format(targeted_class, correct_attack/correct*100, different_layer/total_layer*100))
    return num_all, correct_all, correct_attack_all, total_layer_all, different_layer_all


if __name__ == '__main__':
    args = parse_args()
    now_datetime = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    file_name = '_'.join(['eps'+str(args.eps), 'iter'+str(args.n_iter), 'lr'+str(args.lr), str(args.loss), 'mask'+str(args.mask_rate)])
    if args.t:
        file_name += '_t'
    if args.m:
        file_name += '_m'
    if args.pgd:
        file_name += '_pgd'
    if args.nes:
        file_name += '_nes'
    if args.adam:
        file_name += '_adam'
    if args.auto_lr:
        file_name += '_autolr'
    if args.exp is not None:
        file_name += '_' + args.exp
    if args.model != 'resnet':
        save_path = args.save_path = os.path.join('outputs', args.model, args.dataset, args.type, file_name)
    else:
        args.type = 'fgm'
        save_path = args.save_path = os.path.join('outputs', args.model, args.dataset, 'resnet' + args.resnet, args.type, file_name)
    if os.path.exists(os.path.join(save_path, 'log.txt')):
        os.remove(os.path.join(save_path, 'log.txt'))
    if os.path.exists(os.path.join(save_path, 'out.pickle')):
        os.remove(os.path.join(save_path, 'out.pickle'))
    os.makedirs(save_path, exist_ok=True)
    
    args.logger_file = os.path.join(save_path, 'log.txt')
    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    if args.model == 'skipnet':
        args.log_head = 'SkipNet'
    elif args.model == 'dynconv':
        args.log_head = 'DynConv'
    elif args.model == 'resnet':
        args.log_head = 'ResNet'
    if args.type == 'lgm':
        args.log_head += '-LGM:'
    else:
        args.log_head += '-FGM:'

    if args.dataset == 'cifar':
        args.image_size = 32
        args.class_num = 10
    elif args.dataset == 'imagenet':
        args.image_size = 224
        args.class_num = 1000

    if args.t:
        num_all = 0
        correct_all = 0
        correct_attack_all = 0
        total_layer_all = 0
        different_layer_all = 0
        if isinstance(args.lamb, list):
            lamb_list = args.lamb
        else:
            lamb_list = None
        for targeted_class in range(args.class_num):
            if lamb_list is not None:
                args.lamb = lamb_list[targeted_class]
            num, correct, correct_attack, total_layer, different_layer = attack_process(args, targeted_class)
            num_all += num
            correct_all += correct
            correct_attack_all += correct_attack
            total_layer_all += total_layer
            different_layer_all += different_layer
        log_str = args.log_head + '\tNum:{}\t' 'Correct:{}\t' 'Correct Attack:{}\t' 'Attack Success Rate:{:.2f}%\t'.format(num_all, correct_all, correct_attack_all, correct_attack_all/correct_all*100)
    else:
        num_all, correct_all, correct_attack_all, total_layer_all, different_layer_all = attack_process(args)
        log_str = args.log_head + '\tNum:{}\t' 'Model ACC:{:.2f}%\t' 'ACC after Attack:{:.2f}%\t' 'Attack Success Rate:{:.2f}%\t'.format(num_all, correct_all/num_all*100, correct_attack_all/num_all*100, (correct_all-correct_attack_all)/correct_all*100)
    if not args.model == 'resnet':
        log_str += 'Different Layer Rate:{:.2f}%'.format(different_layer_all/total_layer_all*100)
    logging.info(log_str)
    
