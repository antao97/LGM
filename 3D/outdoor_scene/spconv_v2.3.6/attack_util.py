''' Attack utilities

Dynamics-aware Adversarial Attack of Adaptive Neural Networks

Author: An Tao, Pengliang Ji
Email: ta19@mails.tsinghua.edu.cn, jpl1723@buaa.edu.cn
Date: 2023/6/15

'''

import os
import numpy as np
from config.config import load_config_data
import torch
from dataloader.pc_dataset import get_pc_model_class
from dataloader.dataset_semantickitti import nb_process_label
from utils.metric_util import per_class_iu, fast_hist_crop
from torch_scatter import scatter_mean, scatter_sum, scatter_mul, scatter_max


config_path = './config/semantickitti.yaml'
configs = load_config_data(config_path)
train_dataloader_config = configs['train_data_loader']
val_dataloader_config = configs['val_data_loader']
data_path = train_dataloader_config["data_path"]
dataset_config = configs['dataset_params']
label_mapping = dataset_config["label_mapping"]
SemKITTI = get_pc_model_class(dataset_config['pc_dataset_type'])
val_imageset = val_dataloader_config["imageset"]
val_ref = val_dataloader_config["return_ref"]
nusc=None
val_pt_dataset = SemKITTI(data_path, imageset=val_imageset, return_ref=val_ref, label_mapping=label_mapping, nusc=nusc)

model_config = configs['model_params']
grid_size = torch.Tensor(model_config['output_shape'])
max_volume_space = [50.0, 3.1415926, 2.0]
min_volume_space = [0.0, -3.1415926, -4.0]
max_bound = torch.Tensor(max_volume_space)
min_bound = torch.Tensor(min_volume_space)
crop_range = max_bound - min_bound
cur_grid_size = grid_size
intervals = crop_range / (cur_grid_size - 1)

num_class = model_config['num_class']
    

def cart2polar(input_xyz, input_np=True):
    if input_np:
        rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
        phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
        return np.stack((rho, phi, input_xyz[:, 2]), axis=1)
    else:
        rho = torch.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
        phi = torch.atan2(input_xyz[:, 1], input_xyz[:, 0])
        return torch.stack((rho, phi, input_xyz[:, 2]), axis=1)

def polar2cat(input_xyz_polar, input_np=True):
    if input_np:
        x = input_xyz_polar[:, 0] * np.cos(input_xyz_polar[:, 1])
        y = input_xyz_polar[:, 0] * np.sin(input_xyz_polar[:, 1])
        return np.stack((x, y, input_xyz_polar[:, 2]), axis=1)
    else:
        x = input_xyz_polar[:, 0] * torch.cos(input_xyz_polar[:, 1])
        y = input_xyz_polar[:, 0] * torch.sin(input_xyz_polar[:, 1])
        return torch.stack((x, y, input_xyz_polar[:, 2]), axis=1)

def quantize(xyz, sig=None):
    if len(sig.shape) == 2: sig = torch.squeeze(sig)
    xyz_pol = cart2polar(xyz, input_np=False)
    grid_ind = []
    for i in range(3):
        grid_ind.append(torch.floor((torch.clip(xyz_pol[:,i], min_bound[i], max_bound[i]) - min_bound[i]) / intervals[i]).unsqueeze(1))
    grid_ind = torch.cat(grid_ind, dim=1)
    voxel_centers = (grid_ind + 0.5) * intervals + min_bound
    return_xyz = xyz_pol - voxel_centers
    return_xyz = torch.cat((return_xyz, xyz_pol, xyz[:, :2]), dim=1)
    return_fea = torch.cat((return_xyz, sig.unsqueeze(1)), dim=1)
    return grid_ind, return_fea


def clip_by_tensor(t,t_min,t_max):
    t=t.float()
    t_min=t_min.float()
    t_max=t_max.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


def device_transfer(tensor, device):
    tensor = tensor.to('cuda:'+str(device))
    return tensor

# indices, features, spatial_shape, batch_size
def device_transfer_sparse(sparse_tensor, device):
    sparse_tensor.indices = sparse_tensor.indices.to('cuda:'+str(device))   
    sparse_tensor.features = sparse_tensor.features.to('cuda:'+str(device))
    return sparse_tensor


class IOStream():
    """
    Print logs in file
    """
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def load_file(coords_pcl, sig, requires_grad=True):
    """
    Load point clouds
    """

    if not torch.is_tensor(coords_pcl):
        coords_pcl = torch.from_numpy(coords_pcl).float()
    if not torch.is_tensor(sig):
        sig = torch.from_numpy(sig)
    if requires_grad:
        coords_pcl.requires_grad_(True).retain_grad()
    coords_vox, feats_vox = quantize(coords_pcl, sig)
    # feats = feats_vox[:,3:9]
    # idx, inverse_idx = ME.utils.quantization.quantize(coords_vox)
    # feats_grid = feats_vox[:,3:9]
    coords_vox = coords_vox.detach().long().requires_grad_(False)
    return coords_pcl, coords_vox, feats_vox


def generate_input_sparse_tensor(attack_configs, coords_pcl, sig, extend=True, coords_pcl0=None, requires_grad=True):
    """
    Obtain sparse tensor for input
    """

    coords_pcl, coords_vox, feats = load_file(coords_pcl, sig, requires_grad=requires_grad)
    if extend:
        coords_vox, feats, occupy = add_occupancy(attack_configs, coords_pcl0, coords_pcl, coords_vox, feats)
        return coords_pcl, coords_vox, feats, occupy
    else:
        return coords_pcl, coords_vox, feats


def label_to_grid(grid_ind, labels, ignore_label=0):
    processed_label = np.ones(grid_size.long().tolist(), dtype=np.uint8) * ignore_label
    label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
    label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
    processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
    return processed_label


def add_occupancy(attack_configs, coords_pcl0, coords_pcl, coords_vox_old, feats_old):
    """
    Obtain occupancy values for input voxelization and sparse convolution
    """

    ### Find Valid Voxels ###

    # Find all possible voxels that may be occupied after an attack step.
    coords_vox_all = []
    coords_vox = []
    valid = []
    i = 0
    for dx in [0, -1, 1]:
        for dy in [0, -1, 1]:
            for dz in [0, -1, 1]:
                if i == 0:
                    # Add existing occupied voxels
                    valid.append(torch.arange(coords_pcl.shape[0]))     
                    coords_vox_all.append(coords_vox_old)
                else:
                    # Examine neighbor voxels whether in step size
                    coords_vox_new = coords_vox_old + torch.Tensor([dx, dy, dz])
                    coords_pcl_new1 = polar2cat(coords_vox_new * intervals, input_np=False)
                    coords_pcl_new2 = polar2cat((coords_vox_new + 1) * intervals, input_np=False)
                    valid_new1 = torch.where((coords_pcl - coords_pcl_new1).abs() < attack_configs.step)[0]
                    valid_new2 = torch.where((coords_pcl - coords_pcl_new2).abs() < attack_configs.step)[0]
                    valid_new = torch.unique(torch.cat([valid_new1, valid_new2]))

                    unvalid_new1 = torch.where(coords_vox_new <= 0)[0]
                    unvalid_new2 = torch.where(grid_size <= coords_vox_new)[0]

                    count = torch.zeros(coords_pcl.shape[0])
                    count[valid_new] += 1
                    count[unvalid_new1] -= 1
                    count[unvalid_new2] -= 1
                    valid_new = torch.where(count == 1)[0]

                    valid.append(valid_new)
                    coords_vox_all.append(coords_vox_new)
                coords_vox.append(coords_vox_all[i][valid[i]])
                i = i + 1
    coords_vox = torch.cat(coords_vox, dim=0)


    ### Relation Calculation ###

    # inverse_idx = torch.Tensor(inverse_idx[0]).long()
    relation_input_list = []
    relation_conv_list = []
    coords_grid = cart2polar(coords_pcl, input_np=False)
    i = 0
    for dx in [0, -1, 1]:
        for dy in [0, -1, 1]:
            for dz in [0, -1, 1]:
                # Distance
                clip1 = (coords_vox_all[i][valid[i]] - grid_size + 1) == 0
                clip2 = coords_vox_all[i][valid[i]] == 0
                coords_vox_nei_valid = coords_vox_all[i][valid[i]] + 0.5
                coords_vox_nei_valid[clip1] = coords_vox_all[i][valid[i]][clip1].float()
                coords_vox_nei_valid[clip2] = coords_vox_all[i][valid[i]][clip2].float()
                
                coords_pol_valid = cart2polar(coords_pcl[valid[i]], input_np=False)
                coords_vox_valid = []
                for dim in range(3):
                    coords_vox_valid.append(((torch.clip(coords_pol_valid[:,dim], min_bound[dim], max_bound[dim]) - min_bound[dim]) / intervals[dim]).unsqueeze(1))
                coords_vox_valid = torch.cat(coords_vox_valid, dim=1)
                dist = torch.abs(coords_vox_valid  - coords_vox_nei_valid)

                # Relation for input voxelization
                relation_input = torch.prod(1/(1+torch.exp(attack_configs.lamda_input*(dist-0.5))), dim=-1)
                relation_conv = torch.prod(1/(1+torch.exp(attack_configs.lamda_conv*(dist-0.5))), dim=-1)
                relation_input_list.append(relation_input)
                relation_conv_list.append(relation_conv)

                i = i + 1
    relation_input_list = torch.cat(relation_input_list, dim=0)
    relation_conv_list = torch.cat(relation_conv_list, dim=0)


    ### Gathering Operation ###

    # Obtain neighbor mapping in Equations (10) and (18)
    coords_vox_unique, inverse_mapping = torch.unique(coords_vox.detach(), dim=0, return_inverse=True)
    unique_idx = scatter_max(torch.arange(inverse_mapping.shape[0]), inverse_mapping)[0]
    
    # The gathering function in Equation (10)
    occupy_conv = 1 - scatter_mul(1-relation_conv_list, inverse_mapping)
    occupy_conv = occupy_conv[inverse_mapping]
    

    ### Input Voxelization ###
    
    # Equation (18)
    feats_list = []
    for i in range(len(valid)):
        feats_list.append(feats_old[valid[i]])
    feats_list = torch.cat(feats_list, dim=0)
    mid_result = relation_input_list.unsqueeze(1).repeat(1,feats_old.shape[-1]) * feats_list
    feats_tilde = []
    for i in range(feats_old.shape[-1]):
        feats_tilde.append(scatter_sum(mid_result[:,i], inverse_mapping))
    feats_tilde = torch.stack(feats_tilde, dim=1)
    relation_input_sum = scatter_sum(relation_input_list, inverse_mapping)
    feats_tilde = feats_tilde / (relation_input_sum.unsqueeze(1))

    feats_tilde = feats_tilde[inverse_mapping]
    feats_tilde = torch.cat([feats_old, feats_tilde[feats_old.shape[0]:]])

    # Equation (19)
    feats = occupy_conv.unsqueeze(1).repeat(1,feats_old.shape[-1]) * feats_tilde

    # Return input sparse tensor, occupancy values for sparse convolution in network
    return coords_vox.long(), feats, occupy_conv


def test_IoU(hist_list, all=False):
    iou = per_class_iu(sum(hist_list))
    if all:
        for class_name, class_iou in zip(unique_label_str, iou):
            logging.info(f"{class_name}: {class_iou * 100}")
    val_miou = np.nanmean(iou) * 100
    if all:
        logging.info(f"mIoU: {val_miou}")
    return val_miou


def save_prediction(attack_configs, save_path, room_name, preds, probs=None):
    """
    Save network prediction
    """
    if attack_configs.save_preds:
        if not os.path.exists(os.path.join(save_path, 'pred')):
            os.makedirs(os.path.join(save_path, 'pred'))
        np.savetxt(os.path.join(save_path, 'pred', room_name+'.txt'), preds, fmt='%d')

    if attack_configs.save_probs:
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=6)
        if not os.path.exists(os.path.join(save_path, 'prob')):
            os.makedirs(os.path.join(save_path, 'prob'))
        np.savetxt(os.path.join(save_path, 'prob', room_name+'.txt'), probs, fmt='%.6f')
    return


def save_attacked_coords(save_path, room_name, coords_pcl):
    """
    Save attacked point cloud coordinates
    """
    if not os.path.exists(os.path.join(save_path, 'coord')):
        os.makedirs(os.path.join(save_path, 'coord'))
    np.savetxt(os.path.join(save_path, 'coord', room_name+'.txt'), coords_pcl)
    return


def freeze_bn(model):
    """
    Freeze batch normalization layers as evaluation mode
    """
    model.cylinder_3d_generator.PPmodel[0].eval()
    model.cylinder_3d_generator.PPmodel[2].eval()
    model.cylinder_3d_generator.PPmodel[5].eval()
    model.cylinder_3d_generator.PPmodel[8].eval()

    model.cylinder_3d_spconv_seg.downCntx.bn0.eval()
    model.cylinder_3d_spconv_seg.downCntx.bn0_2.eval()
    model.cylinder_3d_spconv_seg.downCntx.bn1.eval()
    model.cylinder_3d_spconv_seg.downCntx.bn2.eval()

    model.cylinder_3d_spconv_seg.resBlock2.bn0.eval()
    model.cylinder_3d_spconv_seg.resBlock2.bn0_2.eval()
    model.cylinder_3d_spconv_seg.resBlock2.bn1.eval()
    model.cylinder_3d_spconv_seg.resBlock2.bn2.eval()

    model.cylinder_3d_spconv_seg.resBlock3.bn0.eval()
    model.cylinder_3d_spconv_seg.resBlock3.bn0_2.eval()
    model.cylinder_3d_spconv_seg.resBlock3.bn1.eval()
    model.cylinder_3d_spconv_seg.resBlock3.bn2.eval()

    model.cylinder_3d_spconv_seg.resBlock4.bn0.eval()
    model.cylinder_3d_spconv_seg.resBlock4.bn0_2.eval()
    model.cylinder_3d_spconv_seg.resBlock4.bn1.eval()
    model.cylinder_3d_spconv_seg.resBlock4.bn2.eval()

    model.cylinder_3d_spconv_seg.resBlock5.bn0.eval()
    model.cylinder_3d_spconv_seg.resBlock5.bn0_2.eval()
    model.cylinder_3d_spconv_seg.resBlock5.bn1.eval()
    model.cylinder_3d_spconv_seg.resBlock5.bn2.eval()

    model.cylinder_3d_spconv_seg.upBlock0.trans_bn.eval()
    model.cylinder_3d_spconv_seg.upBlock0.bn1.eval()
    model.cylinder_3d_spconv_seg.upBlock0.bn2.eval()
    model.cylinder_3d_spconv_seg.upBlock0.bn3.eval()
    
    model.cylinder_3d_spconv_seg.upBlock1.trans_bn.eval()
    model.cylinder_3d_spconv_seg.upBlock1.bn1.eval()
    model.cylinder_3d_spconv_seg.upBlock1.bn2.eval()
    model.cylinder_3d_spconv_seg.upBlock1.bn3.eval()

    model.cylinder_3d_spconv_seg.upBlock2.trans_bn.eval()
    model.cylinder_3d_spconv_seg.upBlock2.bn1.eval()
    model.cylinder_3d_spconv_seg.upBlock2.bn2.eval()
    model.cylinder_3d_spconv_seg.upBlock2.bn3.eval()

    model.cylinder_3d_spconv_seg.upBlock3.trans_bn.eval()
    model.cylinder_3d_spconv_seg.upBlock3.bn1.eval()
    model.cylinder_3d_spconv_seg.upBlock3.bn2.eval()
    model.cylinder_3d_spconv_seg.upBlock3.bn3.eval()  

    model.cylinder_3d_spconv_seg.ReconNet.bn0.eval()
    model.cylinder_3d_spconv_seg.ReconNet.bn0_2.eval()
    model.cylinder_3d_spconv_seg.ReconNet.bn0_3.eval()

    return model


def disable_model_grad(model):
    """
    Disable gradient calculation of model parameters to save memory
    """
    model.cylinder_3d_generator.PPmodel[1].weight.requires_grad_(False)
    model.cylinder_3d_generator.PPmodel[1].bias.requires_grad_(False)
    model.cylinder_3d_generator.PPmodel[4].weight.requires_grad_(False)
    model.cylinder_3d_generator.PPmodel[4].bias.requires_grad_(False)
    model.cylinder_3d_generator.PPmodel[7].weight.requires_grad_(False)
    model.cylinder_3d_generator.PPmodel[7].bias.requires_grad_(False)
    model.cylinder_3d_generator.PPmodel[10].weight.requires_grad_(False)
    model.cylinder_3d_generator.PPmodel[10].bias.requires_grad_(False)

    model.cylinder_3d_spconv_seg.downCntx.conv1.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.downCntx.conv1_2.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.downCntx.conv2.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.downCntx.conv3.weight.requires_grad_(False)

    model.cylinder_3d_spconv_seg.resBlock2.conv1.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.resBlock2.conv1_2.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.resBlock2.conv2.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.resBlock2.conv3.weight.requires_grad_(False)

    model.cylinder_3d_spconv_seg.resBlock3.conv1.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.resBlock3.conv1_2.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.resBlock3.conv2.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.resBlock3.conv3.weight.requires_grad_(False)

    model.cylinder_3d_spconv_seg.resBlock4.conv1.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.resBlock4.conv1_2.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.resBlock4.conv2.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.resBlock4.conv3.weight.requires_grad_(False)

    model.cylinder_3d_spconv_seg.resBlock5.conv1.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.resBlock5.conv1_2.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.resBlock5.conv2.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.resBlock5.conv3.weight.requires_grad_(False)

    model.cylinder_3d_spconv_seg.upBlock0.trans_dilao.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.upBlock0.conv1.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.upBlock0.conv2.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.upBlock0.conv3.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.upBlock0.up_subm.weight.requires_grad_(False)

    model.cylinder_3d_spconv_seg.upBlock1.trans_dilao.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.upBlock1.conv1.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.upBlock1.conv2.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.upBlock1.conv3.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.upBlock1.up_subm.weight.requires_grad_(False)

    model.cylinder_3d_spconv_seg.upBlock2.trans_dilao.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.upBlock2.conv1.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.upBlock2.conv2.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.upBlock2.conv3.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.upBlock2.up_subm.weight.requires_grad_(False)

    model.cylinder_3d_spconv_seg.upBlock3.trans_dilao.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.upBlock3.conv1.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.upBlock3.conv2.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.upBlock3.conv3.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.upBlock3.up_subm.weight.requires_grad_(False)

    model.cylinder_3d_spconv_seg.ReconNet.conv1.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.ReconNet.conv1_2.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.ReconNet.conv1_3.weight.requires_grad_(False)

    model.cylinder_3d_spconv_seg.logits.weight.requires_grad_(False)
    model.cylinder_3d_spconv_seg.logits.bias.requires_grad_(False)

    return model 