''' Evaluation script for the Stanford dataset (S3DIS)

Dynamics-aware Adversarial Attack of Adaptive Neural Networks

Author: Ziyi Wu, An Tao
Email: dazitu616@gmail.com, ta19@mails.tsinghua.edu.cn 
Date: 2022/1/13

Required Inputs:
    --data_path (str): Data path to the dataset.
    
Important Optional Inputs:
    --attacked_coords (str): Evaluate the model performance with attacked point cloud coordinates.
            The format of path is `outputs/stanford/budget_<your budget>/<your exp name>/coord`.
    --exp_name (str): Assign an experiment name. Default is `Logs_<date>_<time>`.
    --save_preds (store_true): Whether to save the class prediction results. Default is `False`.
    --save_probs (store_true): Whether to save the probability values of classes. Default is `False`.
    --visual (store_true): Whether to save the visualization results in `.ply` files. Default is `False`.

Example Usage: 
    python eval_stanford.py --data_path <data path> 

'''

import os
from datetime import datetime
import numpy as np
np.seterr(divide='ignore',invalid='ignore')

import torch
import MinkowskiEngine as ME
import utils

from config.stanford import parser
from models.res16unet import Res16UNet34


# Set labels for ScanNet dataset
VALID_CLASS_NAMES = utils.STANFORD_VALID_CLASS_NAMES


if __name__ == '__main__':

    ### Initialization ###
    
    config = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config.exp_name is None:
        dt = datetime.now()  
        config.exp_name = 'Logs_' + dt.strftime('%Y-%m-%d_%H-%M-%S')

    save_path = os.path.join('outputs/stanford/eval', 'area_' + config.area, config.exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    io = utils.IOStream(save_path + '/run.log')

    # Define a model and load the weights
    model = Res16UNet34(6, 13, config).to(device)
    model_dict = torch.load(config.weights)
    model.load_state_dict(model_dict['state_dict'])
    model.eval()


    ### Evaluation ###
    
    label_all, pred_all = np.array([]), np.array([])
    config.data_path = os.path.join(config.data_path, 'Area_' + config.area)
    all_rooms = os.listdir(config.data_path)
    all_rooms = [name.split('.')[0] for name in all_rooms]
    room_num = len(all_rooms)
    num_classes = len(VALID_CLASS_NAMES)
    io.cprint('Stanford Class Number: {}'.format(num_classes))

    # Start evaluation for each room
    probs_pcl = None
    for idx, room_name in enumerate(all_rooms):
        with torch.no_grad():
            data = os.path.join(config.data_path, room_name)
            inverse_idx, coords_vol, feats_vol, labels_pcl = \
                utils.generate_input_sparse_tensor_eval(
                    data,
                    voxel_size=config.voxel_size,
                    attacked_coords=config.attacked_coords,
                    dataset='stanford')
            labels_pcl = utils.convert_label_stanford(labels_pcl)

            # Feed-forward pass and get the prediction
            sinput = ME.SparseTensor(feats_vol, coords=coords_vol).to(device)
            soutput = model(sinput)
            preds_vol = soutput.F.max(1)[1].cpu().numpy()
            preds_pcl = preds_vol[inverse_idx]
            if config.save_probs:
                probs_vol = torch.nn.functional.softmax(soutput.F, dim=1).cpu().numpy()
                probs_pcl = probs_vol[inverse_idx]

            intersection, union, target = utils.intersectionAndUnion(
                preds_pcl, labels_pcl, num_classes, 255)
            mIoU = np.nanmean(intersection / union)
            print('Room: {:>2}/{:>2}  |  mIoU: {:.4F}'.format(idx, room_num, mIoU))

            # Save results
            pred_all = np.hstack([pred_all, preds_pcl]) if \
                pred_all.size else preds_pcl
            label_all = np.hstack([label_all, labels_pcl]) if \
                label_all.size else labels_pcl
            torch.cuda.empty_cache()

            if config.save_preds or config.save_probs:
                utils.save_prediction(config, save_path, room_name, preds_pcl, probs_pcl, dataset='stanford')

            # Visualization
            if config.visual:
                if config.attacked_coords is None:
                    utils.visualize(config, room_name, None, preds_pcl, save_path)
                else:
                    utils.visualize(config, room_name, None, preds_pcl, save_path, refine=True)

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
