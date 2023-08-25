''' Dynamics-aware attack script for the SemanticKITTI dataset

Dynamics-aware Adversarial Attack of Adaptive Neural Networks

Author: An Tao, Pengliang Ji
Email: ta19@mails.tsinghua.edu.cn, jpl1723@buaa.edu.cn
Date: 2022/1/13

''' 

import os
import argparse
import sys
from datetime import datetime
from torch.cuda import device_count
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data
from utils.metric_util import per_class_iu, fast_hist_crop
from utils.load_save_util import load_checkpoint
from dataloader.pc_dataset import get_pc_model_class
from dataloader.dataset_semantickitti import cylinder_dataset

import warnings
import attack_util
from config.attack import attack_parser, attack_parameters
warnings.filterwarnings("ignore")

attack_configs = attack_parser.parse_args()

if attack_configs.exp_name is None:
    dt = datetime.now()  
    attack_configs.exp_name = 'Logs_' + dt.strftime('%Y-%m-%d_%H-%M-%S')

save_path = os.path.join('outputs', 'budget_' + str(attack_configs.budget), attack_configs.exp_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)
io = attack_util.IOStream(save_path + '/run.log')

# Load parameters for attack
if attack_configs.default_para == True:
    if attack_configs.budget not in [0.005, 0.01, 0.02, 0.05]:
        io.cprint('Cannot load default attack parameters for budget ' + str(attack_configs.budget))
    else:
        if attack_configs.iter_num is None:
            attack_configs.iter_num = attack_parameters[attack_configs.budget]['iter_num']
        if attack_configs.step is None:
            attack_configs.step = attack_parameters[attack_configs.budget]['step']
        if attack_configs.lamda_input is None:
            attack_configs.lamda_input = attack_parameters[attack_configs.budget]['lamda_input']
        if attack_configs.lamda_conv is None:
            attack_configs.lamda_conv = attack_parameters[attack_configs.budget]['lamda_conv']
        io.cprint('Successfully load default attack parameters for budget {}'.format(attack_configs.budget))

# Check whether lack attack parameters  
lack_para = False
if attack_configs.iter_num is None:
    io.cprint('Please give iteration number with --iter_num')
    lack_para = True
if attack_configs.step is None:
    io.cprint('Please give step size with --step')
    lack_para = True
if attack_configs.lamda_input is None:
    io.cprint('Please give lamda for input with --lamda_input')
    lack_para = True
if attack_configs.lamda_conv is None:
    io.cprint('Please give lamda for sparse convolution with --lamda_conv')
    lack_para = True
if lack_para:
    exit(1)

io.cprint(str(attack_configs))

device = torch.device('cuda:0')

config_path = 'config/semantickitti.yaml'
configs = load_config_data(config_path)
train_dataloader_config = configs['train_data_loader']
data_path = train_dataloader_config["data_path"]
configs = load_config_data(config_path)
dataset_config = configs['dataset_params']
train_dataloader_config = configs['train_data_loader']
val_dataloader_config = configs['val_data_loader']
val_batch_size = val_dataloader_config['batch_size']
val_imageset = val_dataloader_config["imageset"]
val_ref = val_dataloader_config["return_ref"]
label_mapping = dataset_config["label_mapping"]
model_config = configs['model_params']
train_hypers = configs['train_params']
grid_size = np.array(model_config['output_shape'])
num_class = model_config['num_class']
ignore_label=dataset_config["ignore_label"]
model_load_path = train_hypers['model_load_path']
SemKITTI = get_pc_model_class(dataset_config['pc_dataset_type'])
SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
output_shape = model_config['output_shape']
num_input_features = model_config['num_input_features']
use_norm = model_config['use_norm']
init_size = model_config['init_size']
fea_dim = model_config['fea_dim']
out_fea_dim = model_config['out_fea_dim']
unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]
model, model_new = model_builder.build(model_config)
model = load_checkpoint(model_load_path, model)
model.to(device)
model.eval()
model_new = load_checkpoint(model_load_path, model_new)
model_new.to(device)
model_new.eval()
nusc=None
loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True, num_class=num_class, ignore_label=ignore_label)
val_pt_dataset = SemKITTI(data_path, imageset=val_imageset, return_ref=val_ref, label_mapping=label_mapping, nusc=nusc)

max_volume_space = [50.0, 3.1415926, 2.0]
min_volume_space = [0.0, -3.1415926, -4.0]
max_bound = np.asarray(max_volume_space)
min_bound = np.asarray(min_volume_space)
crop_range = max_bound - min_bound
cur_grid_size = grid_size
intervals = crop_range / (cur_grid_size - 1)


### Attack ###

# Note:
#   Because the perturbed point cloud finally needs to be tested in original sparse convolution network,
#   we test the performance of the original network during our attack at each iteration.

io.cprint("Length of Test Dataset: {}".format(len(val_pt_dataset)))
io.cprint("Class Num: {}".format(num_class))
mIoU0_all = 66.91
hist_list_all_orig = []
hist_list_all = []
for scene_iter, (coords_pcl, labels, sig) in enumerate(val_pt_dataset):
    coords_pcl0 = torch.from_numpy(coords_pcl).clone()
    coords_pcl_best = None
    coords_vox_best = None
    mIoU_best_orig = 100
    mIoU_best = 100
    hist_list_best_orig = []
    hist_list_best = []
    preds_pcl_orig_best = None
    probs_pcl_orig_best = None

    load_attacked_coords = False
    if attack_configs.resume_path is not None:
        attacked_coords_path = os.path.join(attack_configs.resume_path, 'coord', str(scene_iter) + '.txt')
        if os.path.exists(attacked_coords_path):
            coords_pcl = np.loadtxt(attacked_coords_path)
            load_attacked_coords = True

    for iter in range(attack_configs.iter_num):
        hist_list_orig = []
        hist_list = []
        
        # Obtain performance on original sparse convolution network
        with torch.no_grad():
            coords_pcl, coords_vox_orig, feats_orig = \
                attack_util.generate_input_sparse_tensor(
                    attack_configs, 
                    coords_pcl=coords_pcl, 
                    sig=sig,
                    extend=False,
                    requires_grad=False)
            coords_vox_orig = coords_vox_orig.to(device)
            feats_orig = feats_orig.to(device)
            
            pdb.set_trace()
            outputs_orig = model([feats_orig], [coords_vox_orig], val_batch_size)
            predict_labels_orig_npy = torch.argmax(outputs_orig, dim=1).cpu().detach().numpy()
            coords_vox_orig = coords_vox_orig.cpu()
            coords_vox_orig_npy = coords_vox_orig.numpy()
            hist_list_orig.append(fast_hist_crop(predict_labels_orig_npy[0, coords_vox_orig_npy[:, 0], coords_vox_orig_npy[:, 1], coords_vox_orig_npy[:, 2]], labels, unique_label))
            iou_orig = attack_util.test_IoU(hist_list_orig)
            if iter == 0:
                mIoU0 = iou_orig
                coords_pcl_best = coords_pcl.cpu().detach().numpy()
                coords_vox_best = coords_vox_orig.cpu().detach().numpy()
                preds_pcl_orig_best = predict_labels_orig_npy
                probs_pcl_orig_best = outputs_orig.cpu().detach().numpy()
                hist_list_best_orig = hist_list_orig
            torch.cuda.empty_cache()

        if load_attacked_coords:
            break


        # Obtain performance on our modified dynamics-aware sparse convolution network
        coords_pcl, coords_vox, feats, occupy = \
            attack_util.generate_input_sparse_tensor(
                attack_configs, 
                coords_pcl=coords_pcl,
                sig=sig,
                extend=True,
                coords_pcl0=coords_pcl0,
                requires_grad=True)
        feats = feats.type(torch.FloatTensor).to(device)
        coords_vox = coords_vox.to(device)
        occupy = occupy.unsqueeze(1).to(device)
        outputs = model_new([feats], [coords_vox], val_batch_size, occupy=occupy)
        
        predict_labels = torch.argmax(outputs, dim=1)
        predict_labels_npy = predict_labels.cpu().detach().numpy()
        coords_vox = coords_vox.cpu().numpy()
        hist_list.append(fast_hist_crop(predict_labels_npy[0, coords_vox_orig_npy[:, 0], coords_vox_orig_npy[:, 1], coords_vox_orig_npy[:, 2]], labels, unique_label))
        iou = attack_util.test_IoU(hist_list)
        
        labels_grid = attack_util.label_to_grid(coords_vox_orig_npy, labels)
        labels_grid = torch.from_numpy(labels_grid).type(torch.LongTensor).to(device)
        loss = -lovasz_softmax(torch.nn.functional.softmax(outputs), labels_grid, ignore=0)
        model.zero_grad()
        model_new.zero_grad()
        loss.backward()

        if iou_orig < mIoU_best_orig:
            mIoU_best_orig = iou_orig
            mIoU_best = iou
            hist_list_best_orig = hist_list_orig
            coords_pcl_best = coords_pcl.cpu().detach().numpy()
            coords_vox_best = coords_vox_orig.cpu().detach().numpy()
            preds_pcl_orig_best = predict_labels_orig_npy
            probs_pcl_orig_best = outputs_orig.cpu().detach().numpy()

        # Perturb the point cloud
        if iter != (attack_configs.iter_num - 1):
            coords_pcl_grad = coords_pcl.grad
            coords_pcl_grad[torch.isnan(coords_pcl_grad)] = 0
            coords_pcl = coords_pcl - attack_configs.step * (coords_pcl_grad / (torch.max(torch.abs(coords_pcl_grad), dim=-1)[0].unsqueeze(1).repeat(1, 3) + 1e-12))
            coords_pcl = torch.where(coords_pcl < (coords_pcl0 - attack_configs.budget), coords_pcl0 - attack_configs.budget, coords_pcl)
            coords_pcl = torch.where(coords_pcl > (coords_pcl0 + attack_configs.budget), coords_pcl0 + attack_configs.budget, coords_pcl)
            coords_pcl = coords_pcl.cpu().detach()
        torch.cuda.empty_cache()
        io.cprint('Scene: {:>4}/{:>4}  |  Iter: {:>2}/{:>2}  |  mIoU: [Original Conv] {:.4F}, [Dyn-aware Conv] {:.4F}'\
            .format(scene_iter, len(val_pt_dataset), iter, attack_configs.iter_num, iou_orig, iou))

    if load_attacked_coords:
        io.cprint('=> Resume Scene: {:>4}/{:>4}  Attacked mIoU: [Original Conv] {:.4F}'.format(scene_iter, len(val_pt_dataset), iou_orig))
    else:
        io.cprint('=> Attack Finished!  mIoU: [Original Conv] {:.4F} -> {:.4F}'.format(mIoU0, mIoU_best_orig))

    hist_list_all_orig.extend(hist_list_best_orig)
    mIoU_temp = attack_util.test_IoU(hist_list_all_orig)
    io.cprint('=> AVG mIoU: [Original Conv] {:.4F} \n'.format(mIoU_temp))

    if attack_configs.save_preds or attack_configs.save_probs:
        preds = preds_pcl_orig_best[0, coords_vox_best[:,0], coords_vox_best[:,1], coords_vox_best[:,2]]
        probs = probs_pcl_orig_best[0, :, coords_vox_best[:,0], coords_vox_best[:,1], coords_vox_best[:,2]]
        attack_util.save_prediction(attack_configs, save_path, str(scene_iter), preds, probs)

    if attack_configs.save_coords and (load_attacked_coords == False):
        attack_util.save_attacked_coords(save_path, str(scene_iter), coords_pcl_best)

mIoU_all_orig = attack_util.test_IoU(hist_list_all_orig)
io.cprint("Final mIoU {:.4F}".format(mIoU_all_orig))
io.cprint('\n' + str(attack_configs))
io.close()
