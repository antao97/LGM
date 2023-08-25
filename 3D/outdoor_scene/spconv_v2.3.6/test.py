''' Evaluation script for the SemanticKITTI dataset

Dynamics-aware Adversarial Attack of Adaptive Neural Networks

Author: An Tao, Pengliang Ji
Email: ta19@mails.tsinghua.edu.cn, jpl1723@buaa.edu.cn
Date: 2023/6/20

''' 

import os
import time
import argparse
import sys
import numpy as np
import yaml
import torch
import torch.optim as optim
from plyfile import PlyData, PlyElement
from tqdm import tqdm
from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name, get_pc_model_class
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data
from utils.load_save_util import load_checkpoint
import attack_util
import warnings
warnings.filterwarnings("ignore")

label_mapping = 'config/label_mapping/semantic-kitti.yaml'
with open(label_mapping, 'r') as stream:
    semkittiyaml = yaml.safe_load(stream)


def visualize(scene_name, coords_pcl, labels_pcl, save_path, remark=None):
    """
    Function for visualization
    """
    rgb = np.array([semkittiyaml['color_map'][semkittiyaml['learning_map_inv'][l]] for l in labels_pcl.reshape(-1)])
    xyzRGB = [(coords_pcl[i, 0], coords_pcl[i, 1], coords_pcl[i, 2], rgb[i, 0], rgb[i, 1], rgb[i, 2]) for i in range(coords_pcl.shape[0])]
    vertex = PlyElement.describe(np.array(xyzRGB, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                
    save_path = os.path.join(save_path, 'visual')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filepath = os.path.join(save_path, scene_name + '_' + remark + '.ply')
    PlyData([vertex]).write(filepath)
    print('PLY visualization file saved in', filepath)


def main(args):
    pytorch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = args.config_path
    configs = load_config_data(config_path)
    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    data_path = train_dataloader_config["data_path"]
    val_dataloader_config = configs['val_data_loader']
    val_batch_size = val_dataloader_config['batch_size']
    val_imageset = val_dataloader_config["imageset"]
    val_ref = val_dataloader_config["return_ref"]
    nusc=None
    train_batch_size = train_dataloader_config['batch_size']
    model_config = configs['model_params']
    train_hypers = configs['train_params']
    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']
    model_load_path = train_hypers['model_load_path']
    model_save_path = train_hypers['model_save_path']
    SemKITTI = get_pc_model_class(dataset_config['pc_dataset_type'])
    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]
    model = model_builder.build(model_config)
    if os.path.exists(model_load_path):
        model = load_checkpoint(model_load_path, model)
    model.to(pytorch_device)
    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                   num_class=num_class, ignore_label=ignore_label)
    val_pt_dataset = SemKITTI(data_path, imageset=val_imageset, return_ref=val_ref, label_mapping=label_mapping, nusc=nusc)

    visual_list = args.visual.split(',')
    if 'all' not in visual_list:
        visual_list = [int(i) for i in visual_list]

    model.eval()
    hist_list = []
    print("Length of test dataset: ", len(val_pt_dataset))
    print("Class num: ", num_class)
    with torch.no_grad():
        for scene_iter, (coords_pcl, labels_pcl, sig) in enumerate(val_pt_dataset):
            load_attacked_coords = False
            if args.coords_path is not None:
                attacked_coords_path = os.path.join(args.coords_path, 'coord', str(scene_iter) + '.txt')
                if os.path.exists(attacked_coords_path):
                    coords_pcl = np.loadtxt(attacked_coords_path)
                    load_attacked_coords = True

            _, coords_vox, feats = \
            attack_util.generate_input_sparse_tensor(
                attack_configs=None, 
                coords_pcl=coords_pcl, 
                sig=sig,
                extend=False,
                requires_grad=True)
            coords_vox = coords_vox.to(pytorch_device)
            feats = feats.to(pytorch_device)
            outputs = model([feats], [coords_vox.detach()], val_batch_size)
            predict_labels = torch.argmax(outputs, dim=1).cpu().detach().numpy()
            coords_vox_npy = coords_vox.cpu().numpy()
            labels_pcl_pred = predict_labels[0, coords_vox_npy[:, 0], coords_vox_npy[:, 1], coords_vox_npy[:, 2]]
            hist_list.append(fast_hist_crop(labels_pcl_pred, labels_pcl, unique_label))
            iou = attack_util.test_IoU(hist_list)
            print('=> Scene: {:>4}/{:>4}  mIoU: {:.4F}'.format(scene_iter, len(val_pt_dataset), iou))
            
            if args.visual is not None:
                if args.visual == 'all' or scene_iter in visual_list:
                    if not load_attacked_coords:
                        visualize(str(scene_iter), coords_pcl, labels_pcl, args.save_path, remark='gt')
                        visualize(str(scene_iter), coords_pcl, labels_pcl_pred, args.save_path, remark='pred')
                    else:
                        visualize(str(scene_iter), coords_pcl, labels_pcl_pred, args.save_path, remark='adv')


    iou = per_class_iu(sum(hist_list))
    print('\nValidation per class iou: ')
    for class_name, class_iou in zip(unique_label_str, iou):
        print('%s : %.2f%%' % (class_name, class_iou * 100))
    val_miou = np.nanmean(iou) * 100
    print('mIoU: ',val_miou)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti.yaml')
    parser.add_argument('-v', '--visual', type=str, default=None, help='Files to visualize')
    parser.add_argument('--coords_path', type=str, default=None,
                    help='Load the coordinates of point cloud data.')
    parser.add_argument('--save_path', type=str, default='./',
                    help='Path to save visualized data.')
    args = parser.parse_args()
    print(' '.join(sys.argv))
    print(args)
    main(args)

