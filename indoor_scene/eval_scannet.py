"""

The Evaluation File for 3D Sparse Convolution Network

@Author: 
    Ziyi Wu,
    An Tao

@Contact: 
    dazitu616@gmail.com, 
    ta19@mails.tsinghua.edu.cn
    
@Time: 
    2022/1/23 9:32 PM

"""

import os
from datetime import datetime
import numpy as np
np.seterr(divide='ignore',invalid='ignore')

import torch
import MinkowskiEngine as ME
import utils

from config.scannet import parser
from models.res16unet import Res16UNet34C

# Set labels for ScanNet dataset
VALID_CLASS_NAMES = utils.SCANNET_VALID_CLASS_NAMES


def load_file(file_name, voxel_size, attacked_coords):
    """
    Load point clouds
    """
    plydata = PlyData.read(file_name+'.ply')
    data = plydata.elements[0].data
    if attacked_coords is not None:
        room_name = file_name.split('/')[-1]
        coords = np.loadtxt(os.path.join(attacked_coords, room_name + '.txt'))
    else:
        coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
    colors = np.array([data['red'], data['green'],
                       data['blue']], dtype=np.float32).T / 255.
    labels = np.array(data['label'], dtype=np.int32)

    feats = colors - 0.5

    idx, inverse_idx, quan_coords, quan_feats = utils.sparse_quantize(
        coords, feats, None, return_index=True,
        return_inverse=True, quantization_size=voxel_size)

    return inverse_idx, quan_coords, quan_feats, labels


def generate_input_sparse_tensor(file_name, voxel_size=0.02, attacked_coords=None):
    """
    Obtain sparse tensor for input
    """

    # Create a batch, this process is done in a data loader during training in parallel.
    batch = [load_file(file_name, voxel_size, attacked_coords)]
    inverse_idx, coordinates_, featrues_, labels = list(zip(*batch))
    coordinates, features = ME.utils.sparse_collate(
        coordinates_, featrues_, None)

    return inverse_idx, coordinates, features.float(), labels[0]


if __name__ == '__main__':

    ### Initialization ###
    
    config = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config.exp_name is None:
        dt = datetime.now()  
        config.exp_name = 'Logs_' + dt.strftime('%Y-%m-%d_%H-%M-%S')

    save_path = os.path.join('outputs/scannet/eval', config.split, config.exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    io = utils.IOStream(save_path + '/run.log')

    # Define a model and load the weights
    model = Res16UNet34C(3, 20, config).to(device)
    model_dict = torch.load(config.weights)
    model.load_state_dict(model_dict['state_dict'])
    model.eval()


    ### Evaluation ###
    
    label_all, pred_all = np.array([]), np.array([])
    if config.split == 'val':
        with open(os.path.join(config.data_path, 'scannetv2_val.txt'), 'r') as f:
            all_rooms = f.readlines()
    elif config.split == 'train':
        with open(os.path.join(config.data_path, 'scannetv2_train.txt'), 'r') as f:
            all_rooms = f.readlines()
    all_rooms = [room[:-1] for room in all_rooms]
    room_num = len(all_rooms)
    num_classes = len(VALID_CLASS_NAMES)
    io.cprint('ScanNet Class Number: {}'.format(num_classes))

    # Start evaluation for each room
    probs_pcl = None
    for idx, room_name in enumerate(all_rooms):
        with torch.no_grad():
            data = os.path.join(config.data_path, room_name)
            inverse_idx, coords_vox, feats_vox, labels_pcl = \
                utils.generate_input_sparse_tensor_eval(
                    data,
                    voxel_size=config.voxel_size,
                    attacked_coords=config.attacked_coords,
                    dataset='scannet')
            labels_pcl = utils.convert_label_scannet(labels_pcl)

            # Feed-forward pass and get the prediction
            sinput = ME.SparseTensor(feats_vox, coords=coords_vox).to(device)
            soutput = model(sinput)
            preds_vox = soutput.F.max(1)[1].cpu().numpy()
            preds_pcl = preds_vox[inverse_idx]
            if config.save_probs:
                probs_vox = torch.nn.functional.softmax(soutput.F, dim=1).cpu().numpy()
                probs_pcl = probs_vox[inverse_idx]

            intersection, union, target = utils.intersectionAndUnion(
                preds_pcl, labels_pcl, num_classes, 255)
            mIoU = np.nanmean(intersection / union)
            print('Room: {:>3}/{:>3}  |  mIoU: {:.4F}'.format(idx, room_num, mIoU))

            # Save results
            pred_all = np.hstack([pred_all, preds_pcl]) if \
                pred_all.size else preds_pcl
            label_all = np.hstack([label_all, labels_pcl]) if \
                label_all.size else labels_pcl
            torch.cuda.empty_cache()

            if config.save_preds or config.save_probs:
                utils.save_prediction(config, save_path, room_name, preds_pcl, probs_pcl, dataset='scannet')

            # Visualization
            if config.visual:
                utils.visualize(config, room_name, None, preds_pcl, save_path)

    intersection, union, target = \
        utils.intersectionAndUnion(
            pred_all, label_all, num_classes, 255)
    iou_class = intersection / (union + 1e-10)
    accuracy_class = intersection / (target + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection) / (sum(target) + 1e-10)
    io.cprint('Evaluation Result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.
          format(mIoU, mAcc, allAcc))
    for i in range(num_classes):
        io.cprint('Class_{} Result: IoU/Accuracy {:.4f}/{:.4f}, Name: {}.'.
              format(i, iou_class[i], accuracy_class[i], VALID_CLASS_NAMES[i]))
