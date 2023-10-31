''' Utility tools

Dynamics-aware Adversarial Attack of Adaptive Neural Networks

Author: An Tao, Yingqi Wang
Email: ta19@mails.tsinghua.edu.cn, yingqi-w19@mails.tsinghua.edu.cn
Date: 2023/10/31

'''

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import logging
import pickle

from config import *


def load_model(args):
    if args.dataset == 'cifar':
        if args.model == 'skipnet':
            if args.resnet is None:
                args.resnet = '110'
            args.model_path = 'skipnet/checkpoints/resnet-' + args.resnet + '-rnn-cifar10.pth.tar'
            args.model_type = 'cifar10_rnn_gate_rl_' + args.resnet
            import skipnet.cifar.models as models
            import skipnet.cifar.models_adv as models_adv
        elif args.model == 'dynconv':
            if args.resnet is None:
                args.resnet = '32'
            args.model_path = 'dynconv/exp/cifar/resnet' + args.resnet + '/sparse05/checkpoint_best.pth'
            args.model_type = 'resnet' + args.resnet
            import dynconv.models as models
        elif args.model == 'resnet':
            args.model_path = 'skipnet/cifar/pytorch_resnet_cifar10/pretrained_models/' + cifar_resnet_dict[args.resnet]
            args.model_type = 'resnet' + args.resnet
            import skipnet.cifar.pytorch_resnet_cifar10.resnet as models
        args.image_size = 32
        args.class_num = 10
        
    elif args.dataset == 'imagenet':
        if args.model == 'skipnet':
            if args.resnet is None:
                args.resnet = '101'
            args.model_path = 'skipnet/checkpoints/resnet-' + args.resnet + '-rnn-imagenet.pth.tar'
            args.model_type = 'imagenet_rnn_gate_rl_' + args.resnet
            import skipnet.imagenet.models as models
            import skipnet.imagenet.models_adv as models_adv
        elif args.model == 'dynconv':
            if args.resnet is None:
                args.resnet = '101'
            args.model_path = 'dynconv/exp/imagenet/resnet' + args.resnet + '/sparse05/checkpoint_best.pth'
            args.model_type = 'resnet' + args.resnet
            import dynconv.models as models
        elif args.model == 'resnet':
            args.model_path = None
            args.model_type = 'resnet' + args.resnet
            import torchvision.models as models
        args.image_size = 224
        args.class_num = 1000

    # create model
    # original model
    if args.model == 'skipnet':
        model = models.__dict__[args.model_type]().cuda()
        model = nn.DataParallel(model)
    elif args.model == 'dynconv':
        model = models.__dict__[args.model_type](sparse=True).cuda()
    elif args.model == 'resnet':
        if args.dataset == 'cifar':
            model = models.__dict__[args.model_type]().cuda()
            model = nn.DataParallel(model)
        elif args.dataset == 'imagenet':
            model = models.__dict__[args.model_type](pretrained=True).cuda()
            model = nn.DataParallel(model)
    
    # dynamic-aware model
    if args.type == 'lgm':
        if args.model == 'skipnet':
            model_da = models_adv.__dict__[args.model_type](lamb=args.lamb, diff_type=args.diff_type).cuda()
            model_da = nn.DataParallel(model_da)
        elif args.model == 'dynconv':
            model_da = models.__dict__[args.model_type](sparse=True, dynamics_aware=True, lamb=args.lamb, diff_type=args.diff_type).cuda()
    else:
        model_da = None
    
    if args.model_path:
        logging.info('=> loading checkpoint `{}`'.format(args.model_path))
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'])
        if args.type == 'lgm':
            model_da.load_state_dict(checkpoint['state_dict'])
        logging.info('=> loaded checkpoint `{}`'.format(args.model_path))

    model.eval()
    model = NormalizedModel(args, model)
    model = MyModel(args, model)

    if args.type == 'lgm':
        model_da.eval()
        if args.model == 'skipnet':
            model_da.module.control.rnn.train(True)
        model_da = NormalizedModel(args, model_da)
        model_da = MyModel(args, model_da)
    return model, model_da


class NormalizedModel():
    def __init__(self, args, model):
        self.model = model
        self.model_type = args.model
        self.dataset = args.dataset
        if self.dataset == 'cifar':
            self.mean = [0.4914, 0.4822, 0.4465]
            self.std = [0.2023, 0.1994, 0.2010]
        else:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
    def __call__(self, inputs):
        if self.model_type == 'skipnet':
            return self.model(transforms.Normalize(self.mean, self.std)(inputs))
        elif self.model_type == 'dynconv':
            return self.model(transforms.Normalize(self.mean, self.std)(inputs))
        elif self.model_type == 'resnet':
            return self.model(transforms.Normalize(self.mean, self.std)(inputs))

class MyModel():
    def __init__(self, args, model):
        self.model = model
        self.model_type = args.model
        self.dataset = args.dataset
    def __call__(self, inputs):
        if self.model_type == 'skipnet':
            output, _, _ = self.model(inputs)
        elif self.model_type == 'dynconv':
            output, _ = self.model(inputs)
        elif self.model_type == 'resnet':
            output = self.model(inputs)
        return output
        
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.task_loss = nn.CrossEntropyLoss().to(device='cuda')

    def forward(self, output, target):
        l = self.task_loss(output, target) 
        #logger.add('loss_task', l.item())
        return l

criterion = Loss()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ListAverageMeter(object):
    """Computes and stores the average and current values of a list"""
    def __init__(self):
        self.len = 10000  # set up the maximum length
        self.reset()

    def reset(self):
        self.val = [0] * self.len
        self.avg = [0] * self.len
        self.sum = [0] * self.len
        self.count = 0

    def set_len(self, n):
        self.len = n
        self.reset()

    def update(self, vals, n=1):
        assert len(vals) == self.len, 'length of vals not equal to self.len'
        self.val = vals
        for i in range(self.len):
            self.sum[i] += self.val[i] * n
        self.count += n
        for i in range(self.len):
            self.avg[i] = self.sum[i] / self.count
            
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
    
def accuracy_detail(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred)).view(-1)

    correct_k = correct.float().sum(0)
    return (correct_k.mul_(100.0 / batch_size), torch.where(correct==True)[0])

def run_attack(args, model, model_da, i, input, target, batch_num, attacker, accuracy_detail, targeted_class=None):
    if model is None:
        model, model_da = load_model(args)
    num = len(input)
    if args.t:
        target = torch.ones(target.size()).long() * targeted_class
    target = target.cuda()
    input = input.cuda()
    with torch.no_grad():
        if args.model == 'skipnet':
            output, masks_origin, _ = model.model(input)
        elif args.model == 'dynconv':
            output, meta_1 = model.model(input)
        elif args.model == 'resnet':
            output = model(input)
    _, succ_idx_1 = accuracy_detail(output, target)
    if args.t:
        succ_idx_1_ = torch.ones(output.shape[0])
        succ_idx_1_[succ_idx_1] = 0
        succ_idx_1 = torch.where(succ_idx_1_)[0]
    if model_da is not None:
        attack_examples = attacker.attack(model_da, input, target, succ_idx_1, model)
    else:
        attack_examples = attacker.attack(model, input, target, succ_idx_1)
    with torch.no_grad():
        if args.model == 'skipnet':
            output, masks, _ = model.model(attack_examples)
        elif args.model == 'dynconv':
            output, meta_2 = model.model(attack_examples)
        elif args.model == 'resnet':
            output = model(attack_examples)
    _, succ_idx_2 = accuracy_detail(output[succ_idx_1], target[succ_idx_1])
    
    if args.model == 'skipnet':
        masks_origin = torch.vstack(masks_origin).T.cpu().detach()
        masks = torch.vstack(masks).T.cpu().detach()

        different_layer = torch.sum(masks_origin[succ_idx_1]!=masks[succ_idx_1])
        total_layer = masks_origin[succ_idx_1].numel()
    
    elif args.model == 'dynconv':
        different_layer = 0
        total_layer = 0
        for j in range(len(meta_1['masks'])):
            masks = meta_2['masks'][j]['std'].hard
            masks_origin = meta_1['masks'][j]['std'].hard
            masks_dilate = meta_2['masks'][j]['dilate'].hard
            masks_dilate_origin = meta_1['masks'][j]['dilate'].hard
            
            different_layer += torch.sum(masks_origin!=masks)
            different_layer += torch.sum(masks_dilate_origin!=masks_dilate)
            total_layer += masks_origin.numel()
            total_layer += masks_dilate_origin.numel()

    elif args.model == 'resnet':
        different_layer = 0
        total_layer = 1
                
    correct = len(succ_idx_1)
    correct_attack = len(succ_idx_2)
    
    if args.save:
        save_pickle = os.path.join(args.save_path, 'out.pickle')
        with open(save_pickle, 'ab') as f:
            for idx in succ_idx_1:
                pickle.dump({'gt': target[idx].item(), 'prec': output[idx].cpu().detach().numpy(), 'mask_o': masks_origin[idx].numpy(), 'mask': masks[idx].numpy()}, f)
    
    if args.t:
        log_str = args.log_head + '[{}/{}]\tTargeted Class:{}\tAttack Success Rate:{:.2f}%\t'.format(i+1, batch_num, targeted_class, correct_attack/correct*100)
    else:
        log_str = args.log_head + '[{}/{}]\tModel ACC:{:.2f}%\tACC after Attack:{:.2f}%\t'.format(i+1, batch_num, correct/num*100, correct_attack/num*100)
    if not args.model == 'resnet':
        log_str += 'Different Layer Rate:{:.2f}%'.format(different_layer/total_layer*100)
    logging.info(log_str)

    
    return num, correct, correct_attack, different_layer, total_layer


class Attack():
    def __init__(self, n_iter=10, lr=1, eps=8, targeted=False, box=(-1, 1), momentum=1, m=False, adam=False, nes=False, pgd=False, auto_lr=False, loss='ce', mask=None, class_num=None, verbose=False):
        self.n_iter = n_iter
        self.lr_ = lr
        self.lr0 = lr * 2/255
        self.eps_ = eps
        self.eps = eps * 2/255
        self.targeted = targeted
        self.box = box
        self.momentum = momentum
        self.m = m
        self.adam = adam
        self.nes = nes
        self.pgd = pgd
        self.auto_lr = auto_lr
        self.loss = loss
        self.mask_o = mask
        self.class_num = class_num
        self.verbose = verbose
        self.kappa = 0
        self.alpha = 0.75
        self.rho = 0.75
        self.criterion = nn.CrossEntropyLoss().to(device='cuda')

    def attack(self, model, inputs, targets, batch_mask=None, model0=None):
        batch_size = inputs.size(0)
        if self.loss == 'cw':
            targets_onehot = torch.zeros(targets.size() + (self.class_num,)).cuda()
            targets_onehot.scatter_(1, targets.unsqueeze(1), 1.0)
        
        if self.mask_o == None:
            self.mask=1
        else:
            self.mask = self.mask_o[None, None, :].repeat(batch_size, 3, 1, 1).cuda()
            
        if batch_mask == None:
            batch_mask = torch.Tensor([k for k in range(batch_size)])
        
        inputs0 = inputs.data.clone()
        pert = torch.zeros(inputs.size()).cuda()
        if self.pgd:
            t = 2 * torch.rand(inputs.shape).to(inputs.device).detach() - 1
            pert = self.mask * self.eps * torch.ones_like(inputs).detach() * t
        self.lr = self.lr0
        if self.adam:
            pert.requires_grad_(True)
            optimizer = torch.optim.Adam([pert], lr=self.lr)
        grad_old = torch.zeros(inputs.shape).cuda()
        pert_old = torch.zeros(inputs.shape).cuda()
        pert_best = torch.zeros(inputs.shape).cuda()
        count = 0
        loss_count1 = 0
        loss_count2 = 0
        loss_old = 1e10
        loss_min = 1e10
        condition1 = False
        condition2 = False
        for i in range(self.n_iter):
            if self.nes:
                with torch.no_grad():
                    pert[:] += self.lr * self.momentum * grad_old / torch.mean(torch.abs(pert.grad))
            if not self.adam:
                pert = pert.detach()
                pert.requires_grad_(True)
            pert_old = pert.data.clone()
            attack_examples = pert + inputs0
            outputs = model(attack_examples)
            if self.targeted:
                if self.loss == 'ce':
                    loss = self.criterion(outputs[batch_mask], targets[batch_mask])
                elif self.loss == 'cw':
                    loss = torch.max(torch.max((1-targets_onehot)*outputs, dim=1)[0]-torch.sum(targets_onehot*outputs, 1), torch.ones(batch_size).cuda()*self.kappa)
                    loss = torch.sum(loss[batch_mask])
            else:
                if self.loss == 'ce':
                    loss = -self.criterion(outputs[batch_mask], targets[batch_mask])
                elif self.loss == 'cw':
                    loss = torch.max(torch.sum(targets_onehot*outputs, 1)-torch.max((1-targets_onehot)*outputs, dim=1)[0], torch.ones(batch_size).cuda()*self.kappa)
                    loss = torch.sum(loss[batch_mask])
            if self.adam:
                optimizer.zero_grad()
            loss.backward()

            if self.auto_lr:
                count += 1
                if loss < loss_old:
                    loss_count1 += 1
                if loss_count1 < self.rho * count:
                    condition1 = True
                if loss >= loss_min and count != 1:
                    loss_count2 += 1
                    if loss_count2 >= 3:
                        condition2 = True
                if condition1 or condition2:
                    logging.info('Iteration: {}/{}\t''Condition 1: {}\t''Condition 2: {}\t''Count: {}'.format(i+1, self.n_iter, condition1, condition2, count))
                    self.lr /= 2
                    if self.adam:
                        optimizer.param_groups[0]['lr'] /= 2
                    condition1 = False
                    condition2 = False
                    loss_count1 = 0
                    loss_count2 = 0
                    count = 0
                    with torch.no_grad():
                        pert[:] = pert_best
                    grad_old = torch.zeros(inputs.shape).cuda()
                    pert_old = torch.zeros(inputs.shape).cuda()
                    loss_old = 1e10
                    continue

            pert.grad[:] *= self.mask
            if self.m:
                pert.grad = self.momentum * grad_old + pert.grad/torch.mean(torch.abs(pert.grad))
            if self.adam:
                optimizer.step()
            with torch.no_grad():
                if not self.adam:
                    if self.auto_lr:
                        pert_ = pert - self.lr * pert.grad.sign()
                        pert_ *= self.mask
                        pert_ = pert_.max(-torch.ones(pert.shape).cuda() * self.eps)
                        pert_ = pert_.min(torch.ones(pert.shape).cuda() * self.eps)
                        pert_ = pert_.max(self.box[0]-inputs0)
                        pert_ = pert_.min(self.box[1]-inputs0)
                        pert -= self.alpha * (pert - pert_) + (1-self.alpha) * (pert_old - pert)
                    else:
                        pert -= self.lr * pert.grad.sign()
                pert *= self.mask
                pert[:] = pert.max(-torch.ones(pert.shape).cuda() * self.eps)
                pert[:] = pert.min(torch.ones(pert.shape).cuda() * self.eps)
                pert[:] = pert.max(self.box[0]-inputs0)
                pert[:] = pert.min(self.box[1]-inputs0)
            grad_old = pert.grad 
            
            if loss < loss_old:
                loss_min = loss.data.clone()
                pert_best = pert.data.clone()
            loss_old = loss.data.clone()

            if self.verbose and (i+1)%10 == 0:
                with torch.no_grad():
                    if model0 is not None:
                        outputs = model0(pert.detach() + inputs0)[batch_mask]
                    else:
                        outputs = model(pert.detach() + inputs0)[batch_mask]
                    _, pred = outputs.topk(1, 1, True, True)
                    pred = pred.t()
                    correct = pred.eq(targets[batch_mask].view(1, -1).expand_as(pred)).view(-1)
                    correct_k = correct.int().sum(0)
                    log_str = 'Iteration: {}/{}\t''Batch Result: {}/{}\t''Loss: {:.2f}\t'.format(i+1, self.n_iter, correct_k, batch_mask.shape[0], loss.data)
                    if self.auto_lr:
                        log_str += 'Lr: {:.4f}'.format(self.lr)
                    logging.info(log_str)

        return pert.detach() + inputs0
