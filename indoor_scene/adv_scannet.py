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

import os
from datetime import datetime
import numpy as np
np.seterr(divide='ignore',invalid='ignore')
import torch
from torch import nn
import torch.nn.functional as F
import utils
from config.scannet import parser, parameters
from models.res16unet import Res16UNet34C
import models.scannet_model as new_model

# Set labels for ScanNet dataset
VALID_CLASS_NAMES = utils.SCANNET_VALID_CLASS_NAMES


### Initialization ###

config = parser.parse_args()

if config.resume_path is not None:
    config.exp_name = os.path.split(config.resume_path)[-1]
elif config.exp_name is None:
    dt = datetime.now()  
    config.exp_name = 'Logs_' + dt.strftime('%Y-%m-%d_%H-%M-%S')

save_path = os.path.join('outputs/scannet', 'budget_' + str(config.budget), config.exp_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

io = utils.IOStream(save_path + '/run.log')

# Load parameters for attack
if config.default_para == True:
    if config.budget not in [0.005, 0.01, 0.02, 0.05]:
        io.cprint('Cannot load default attack parameters for budget ' + str(config.budget))
    else:
        if config.iter_num is None:
            config.iter_num = parameters[config.budget][config.dynamics_aware]['iter_num']
        if config.step is None:
            config.step = parameters[config.budget][config.dynamics_aware]['step']
        if config.lamda_input is None:
            config.lamda_input = parameters[config.budget][config.dynamics_aware]['lamda_input']
        if config.lamda_conv is None:
            config.lamda_conv = parameters[config.budget][config.dynamics_aware]['lamda_conv']
        if config.lamda_output is None:
            config.lamda_output = parameters[config.budget][config.dynamics_aware]['lamda_output']
        io.cprint('Successfully load default attack parameters for budget {}'.format(config.budget))

# Check whether lack attack parameters  
lack_para = False
if config.iter_num is None:
    io.cprint('Please give iteration number with --iter_num')
    lack_para = True
if config.step is None:
    io.cprint('Please give step size with --step')
    lack_para = True
if config.lamda_input is None:
    io.cprint('Please give lamda for input with --lamda_input')
    lack_para = True
if config.lamda_conv is None:
    io.cprint('Please give lamda for convolution with --lamda_conv')
    lack_para = True
if config.lamda_output is None:
    io.cprint('Please give lamda for output with --lamda_output')
    lack_para = True
if lack_para:
    exit(1)

io.cprint(str(config))

device = torch.device('cuda:0')
num_devices = torch.cuda.device_count()

io.cprint('Use {} GPU cards'.format(num_devices))

# Define a model and load the weights
model = Res16UNet34C(3, 20, config).to(device)
model_dict = torch.load(config.weights, map_location=device)
model.load_state_dict(model_dict['state_dict'])
model.eval()

# Split model layers into different GPU cards
if config.dynamics_aware:
    if num_devices >= 8:
        num_devices_real = 8
        new_model_1 = new_model.NewRes16UNet34C_d8_1(3, 20, config).to(device)
        new_model_1.load_state_dict(model_dict['state_dict'])
        new_model_1.eval()

        with torch.cuda.device('cuda:1'):
            new_model_2 = new_model.NewRes16UNet34C_d8_2(3, 20, config).to('cuda:1')
            new_model_2.load_state_dict(torch.load(config.weights, map_location='cuda:1')['state_dict'])
            new_model_2.eval()

        with torch.cuda.device('cuda:2'):
            new_model_3 = new_model.NewRes16UNet34C_d8_3(3, 20, config).to('cuda:2')
            new_model_3.load_state_dict(torch.load(config.weights, map_location='cuda:2')['state_dict'])
            new_model_3.eval()

        with torch.cuda.device('cuda:3'):
            new_model_4 = new_model.NewRes16UNet34C_d8_4(3, 20, config).to('cuda:3')
            new_model_4.load_state_dict(torch.load(config.weights, map_location='cuda:3')['state_dict'])
            new_model_4.eval()

        with torch.cuda.device('cuda:4'):
            new_model_5 = new_model.NewRes16UNet34C_d8_5(3, 20, config).to('cuda:4')
            new_model_5.load_state_dict(torch.load(config.weights, map_location='cuda:4')['state_dict'])
            new_model_5.eval()

        with torch.cuda.device('cuda:5'):
            new_model_6 = new_model.NewRes16UNet34C_d8_6(3, 20, config).to('cuda:5')
            new_model_6.load_state_dict(torch.load(config.weights, map_location='cuda:5')['state_dict'])
            new_model_6.eval()

        with torch.cuda.device('cuda:6'):
            new_model_7 = new_model.NewRes16UNet34C_d8_7(3, 20, config).to('cuda:6')
            new_model_7.load_state_dict(torch.load(config.weights, map_location='cuda:6')['state_dict'])
            new_model_7.eval()

        with torch.cuda.device('cuda:7'):
            new_model_8 = new_model.NewRes16UNet34C_d8_8(3, 20, config).to('cuda:7')
            new_model_8.load_state_dict(torch.load(config.weights, map_location='cuda:7')['state_dict'])
            new_model_8.eval()

    elif num_devices >= 4:
        num_devices_real = 4
        new_model_1 = new_model.NewRes16UNet34C_d4_1(3, 20, config).to(device)
        new_model_1.load_state_dict(model_dict['state_dict'])
        new_model_1.eval()

        with torch.cuda.device('cuda:1'):
            new_model_2 = new_model.NewRes16UNet34C_d4_2(3, 20, config).to('cuda:1')
            new_model_2.load_state_dict(torch.load(config.weights, map_location='cuda:1')['state_dict'])
            new_model_2.eval()

        with torch.cuda.device('cuda:2'):
            new_model_3 = new_model.NewRes16UNet34C_d4_3(3, 20, config).to('cuda:2')
            new_model_3.load_state_dict(torch.load(config.weights, map_location='cuda:2')['state_dict'])
            new_model_3.eval()

        with torch.cuda.device('cuda:3'):
            new_model_4 = new_model.NewRes16UNet34C_d4_4(3, 20, config).to('cuda:3')
            new_model_4.load_state_dict(torch.load(config.weights, map_location='cuda:3')['state_dict'])
            new_model_4.eval()

    elif num_devices >= 2:
        num_devices_real = 2
        new_model_1 = new_model.NewRes16UNet34C_d2_1(3, 20, config).to(device)
        new_model_1.load_state_dict(model_dict['state_dict'])
        new_model_1.eval()

        with torch.cuda.device('cuda:1'):
            new_model_2 = new_model.NewRes16UNet34C_d2_2(3, 20, config).to('cuda:1')
            new_model_2.load_state_dict(torch.load(config.weights, map_location='cuda:1')['state_dict'])
            new_model_2.eval()

    else:
        num_devices_real = 1
        new_model = new_model.NewRes16UNet34C(3, 20, config).to(device)
        new_model.load_state_dict(model_dict['state_dict'])
        new_model.eval()


### Attack ###

# Note:
#   Because the perturbed point cloud finally needs to be tested in original sparse convolution network,
#   we test the performance of the original network during our attack at each iteration.

labels_pcl_all, preds_pcl_all = np.array([]), np.array([])
with open(os.path.join(config.data_path, 'scannetv2_val.txt'), 'r') as f:
    all_rooms = f.readlines()
all_rooms = [room[:-1] for room in all_rooms]
room_num = len(all_rooms)
num_classes = len(VALID_CLASS_NAMES)
io.cprint('ScanNet Class Number: {}'.format(num_classes))

# Start attack for each room
for i, room_name in enumerate(all_rooms):
    coords_pcl = None
    labels_pcl = None
    probs_pcl_orig_best = None
    coords_pcl_best = None
    mIoU_orig_best = 100

    load_attacked_coords = False
    if config.resume_path is not None:
        attacked_coords_path = os.path.join(config.resume_path, 'coord', room_name + '.txt')
        if os.path.exists(attacked_coords_path):
            coords_pcl = np.loadtxt(attacked_coords_path)
            load_attacked_coords = True

    for iter in range(config.iter_num):
        data = os.path.join(config.data_path, room_name)

        # Obtain performance on original sparse convolution network
        with torch.no_grad():
            data = os.path.join(config.data_path, room_name)
            idx, inverse_idx, coords_pcl, sinput_orig, labels_pcl = \
                utils.generate_input_sparse_tensor(
                    data,
                    config,
                    coords_pcl=coords_pcl,
                    labels_pcl=labels_pcl,
                    extend=False,
                    dataset='scannet')

            if iter == 0:
                labels_pcl = utils.convert_label_scannet(labels_pcl)
                coords_pcl0 = coords_pcl.clone()

            sinput_orig = sinput_orig.to(device)
            soutput_orig = model(sinput_orig)
            preds_vox_orig = soutput_orig.F.max(1)[1].cpu().numpy()
            preds_pcl_orig = preds_vox_orig[inverse_idx]
            if config.save_probs:
                probs_vox_orig = torch.nn.functional.softmax(soutput_orig.F, dim=1).cpu().numpy()
                probs_pcl_orig = probs_vox_orig[inverse_idx]

            intersection, union, target = utils.intersectionAndUnion(
                preds_pcl_orig, labels_pcl, num_classes, 255)
            mIoU_orig = np.nanmean(intersection / union)

            if iter == 0:
                mIoU_orig0 = mIoU_orig
                preds_pcl_orig0 = preds_pcl_orig
                preds_pcl_orig_best = preds_pcl_orig
                if config.save_probs:
                    probs_pcl_orig_best = probs_pcl_orig
                coords_pcl_best = coords_pcl.clone()

            if mIoU_orig < mIoU_orig_best:
                mIoU_orig_best = mIoU_orig
                preds_pcl_orig_best = preds_pcl_orig
                if config.save_probs:
                    probs_pcl_orig_best = probs_pcl_orig
                coords_pcl_best = coords_pcl.clone()

            torch.cuda.empty_cache()

            if load_attacked_coords:
                break


        # Obtain performance on our modified dynamics-aware sparse convolution network
        idx, inverse_idx, coords_vox, coords_pcl, sinput, occupy_conv, valid = \
            utils.generate_input_sparse_tensor(
                data,
                config,
                coords_pcl=coords_pcl,
                labels_pcl=labels_pcl,
                dataset='scannet')

        sinput = sinput.to(device)

        if config.dynamics_aware:
            occupy_conv = occupy_conv.to(device)

            if num_devices >= 8:
                interm_1 = new_model_1(sinput, idx.shape[0], occupy_conv)
                with torch.cuda.device('cuda:1'):
                    interm_2 = new_model_2(interm_1)
                with torch.cuda.device('cuda:2'):
                    interm_3 = new_model_3(interm_2)
                with torch.cuda.device('cuda:3'):
                    interm_4 = new_model_4(interm_3)
                with torch.cuda.device('cuda:4'):
                    interm_5 = new_model_5(interm_4)
                with torch.cuda.device('cuda:5'):
                    interm_6 = new_model_6(interm_5)
                with torch.cuda.device('cuda:6'):
                    interm_7 = new_model_7(interm_6)
                with torch.cuda.device('cuda:7'):
                    soutput = new_model_8(interm_7)

            elif num_devices >= 4:
                interm_1 = new_model_1(sinput, idx.shape[0], occupy_conv)
                with torch.cuda.device('cuda:1'):
                    interm_2 = new_model_2(interm_1)
                with torch.cuda.device('cuda:2'):
                    interm_3 = new_model_3(interm_2)
                with torch.cuda.device('cuda:3'):
                    soutput = new_model_4(interm_3)

            elif num_devices >= 2:
                interm = new_model_1(sinput, idx.shape[0], occupy_conv)
                with torch.cuda.device('cuda:1'):
                    soutput = new_model_2(interm)

            else:
                soutput = new_model(sinput, idx.shape[0], occupy_conv)
        else:
            soutput = model(sinput)
        
        outputs_pcl = utils.get_point_output(config, soutput, inverse_idx, coords_vox, coords_pcl, valid)
        preds_pcl = outputs_pcl.max(1)[1].cpu().numpy()

        if (num_devices > 1) and config.dynamics_aware:
            label_sparse = torch.LongTensor(labels_pcl).to('cuda:'+str(num_devices_real-1))
        else:
            label_sparse = torch.LongTensor(labels_pcl).to(device)

        if iter != (config.iter_num - 1):
            loss = F.cross_entropy(outputs_pcl, label_sparse.long(), ignore_index=255)
            loss.backward()

        intersection, union, target = utils.intersectionAndUnion(
            preds_pcl, labels_pcl, num_classes, 255)
        mIoU = np.nanmean(intersection / union)

        io.cprint('Room: {:>3}/{:>3}  |  Iter: {:>2}/{:>2}  |  mIoU: [Original Conv] {:.4F}, [Dyn-aware Conv] {:.4F}'\
                .format(i, room_num, iter, config.iter_num, mIoU_orig, mIoU))

        # Perturb the point cloud
        if iter != (config.iter_num - 1):
            coords_pcl = coords_pcl + config.step * (coords_pcl.grad / (torch.max(torch.abs(coords_pcl.grad), dim=-1)[0].unsqueeze(1).repeat(1, 3) + 1e-8))
            coords_pcl = torch.where(coords_pcl < (coords_pcl0 - config.budget), coords_pcl0 - config.budget, coords_pcl)
            coords_pcl = torch.where(coords_pcl > (coords_pcl0 + config.budget), coords_pcl0 + config.budget, coords_pcl)

        torch.cuda.empty_cache()

    
    # Attack finished
    if load_attacked_coords:
        io.cprint('=> Resume Room: {:>3}/{:>3}  Attacked mIoU: [Original Conv] {:.4F}\n'.format(i, room_num, mIoU_orig_best))
    else:
        io.cprint('=> Attack Finished!  mIoU: [Original Conv] {:.4F} -> {:.4F}\n'.format(mIoU_orig0, mIoU_orig_best))

    # Save results
    preds_pcl_all = np.hstack([preds_pcl_all, preds_pcl_orig_best]) if \
        preds_pcl_all.size else preds_pcl_orig_best
    labels_pcl_all = np.hstack([labels_pcl_all, labels_pcl]) if \
        labels_pcl_all.size else labels_pcl

    if config.save_preds or config.save_probs:
        utils.save_prediction(config, save_path, room_name, preds_pcl_orig_best, probs_pcl_orig_best, dataset='scannet')

    if config.save_coords and (load_attacked_coords == False):
        utils.save_attacked_coords(save_path, room_name, coords_pcl_best.numpy())

    # Visualization
    if config.visual:
        utils.visualize(config, room_name, coords_pcl0.detach().numpy(), labels_pcl, save_path, remark='gt')
        utils.visualize(config, room_name, coords_pcl0.detach().numpy(), preds_pcl_orig0, save_path, remark='noattack')
        utils.visualize(config, room_name, coords_pcl.detach().numpy(), preds_pcl_orig_best, save_path, remark='attack')

# Calculate final performance
intersection, union, target = \
    utils.intersectionAndUnion(
        preds_pcl_all, labels_pcl_all, num_classes, 255)
iou_class = intersection / (union + 1e-10)
accuracy_class = intersection / (target + 1e-10)
mIoU = np.mean(iou_class)
mAcc = np.mean(accuracy_class)
allAcc = sum(intersection) / (sum(target) + 1e-10)
io.cprint('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.
      format(mIoU, mAcc, allAcc))
for i in range(num_classes):
    io.cprint('Class_{} Result: Iou/Accuracy {:.4f}/{:.4f}, Name: {}.'.
          format(i, iou_class[i], accuracy_class[i], VALID_CLASS_NAMES[i]))

io.cprint('\n' + str(config))
io.close()