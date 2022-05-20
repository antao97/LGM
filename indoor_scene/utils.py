import os
import torch
import numpy as np
from collections import Sequence
from plyfile import PlyData
from torch_scatter import scatter_sum, scatter_mul
import MinkowskiEngineBackend as MEB
import MinkowskiEngine as ME


# Use ScanNet default colors
colors = [
       (0, 0, 0),           # unlabeled 0
       (174, 199, 232),     # wall 1
       (152, 223, 138),     # floor 2
       (31, 119, 180),      # cabinet 3
       (255, 187, 120),     # bed 4
       (188, 189, 34),      # chair 5
       (140, 86, 75),       # sofa 6
       (255, 152, 150),     # table 7
       (214, 39, 40),       # door 8
       (197, 176, 213),     # window 9
       (148, 103, 189),     # bookshelf 10
       (196, 156, 148),     # picture 11
       (23, 190, 207),      # counter 12
       (178, 76, 76),  
       (247, 182, 210),     # desk 14
       (66, 188, 102), 
       (219, 219, 141),     # curtain 16
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14),      # refrigerator 24
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),     # shower curtain 28
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),       # toilet 33
       (112, 128, 144),     # sink 34
       (96, 207, 209), 
       (227, 119, 194),     # bathtub 36
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),       # otherfurn 39
       (100, 85, 144)
    ]


# Set labels for the ScanNet dataset
SCANNET_VALID_CLASS_NAMES = ['Wall', 'Floor', 'Cabinet', 'Bed', 'Chair', 'Sofa', 'Table', 'Door', 'Window', 'Bookshelf', 'Picture',
                            'Counter', 'Desk', 'Curtain', 'Refrigerator', 'Showercurtain', 'Toilet', 'Sink', 'Bathtub', 'Otherfurniture']
SCANNET_VALID_CLASS_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]

# Set labels for the Stanford dataset
STANFORD_VALID_CLASS_NAMES = ['Clutter', 'Beam', 'Board', 'Bookcase', 'Ceiling', 'Chair', 'Column', 'Door', 'Floor', 'Sofa',
                            'Table', 'Wall', 'Window']
STANFORD_VALID_CLASS_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13]


def convert_label_scannet(label): 
    """
    Ignore invalid labels in the ScanNet dataset
    """
    ignore_class_ids = tuple(set(range(41)) - set(SCANNET_VALID_CLASS_IDS))
    for ignore_lbl in ignore_class_ids:
        if ignore_lbl in label:
            label[label == ignore_lbl] = 255
    for i, lbl in enumerate(SCANNET_VALID_CLASS_IDS):
        if lbl in label:
            label[label == lbl] = i
    return label


def convert_label_stanford(label):
    """
    Ignore invalid labels in the Stanford dataset
    """

    # Remove the class 10 'stairs' class.
    label[label == 10] = 255

    for i in [11, 12, 13]:
        label[label == i] = i - 1
    return label


def load_file(file_name, voxel_size, coords_pcl=None, labels_pcl=None, dataset='scannet'):  
    """
    Load point clouds
    """
    
    plydata = PlyData.read(file_name+'.ply')
    data = plydata.elements[0].data
    if coords_pcl is not None:
        coords_pcl = np.array(coords_pcl.data, dtype=np.float32)
    else:
        coords_pcl = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
    colors_pcl = np.array([data['red'], data['green'],
                       data['blue']], dtype=np.float32).T / 255.
    if labels_pcl is None:
        labels_pcl = np.array(data['label'], dtype=np.int32)

    if dataset == 'scannet':
        feats_pcl = colors_pcl - 0.5
    elif dataset == 'stanford':
        # Normalize feature
        coords_vox = np.floor(coords_pcl / voxel_size)
        coords_vox_mean = coords_vox.mean(0)
        coords_vox_mean[-1] = 0.  # only center x, y!
        coords_vox_norm = coords_vox - coords_vox_mean
        feats_pcl = np.concatenate((colors_pcl - 0.5, coords_vox_norm), 1)

    idx, inverse_idx, coords_vox, feats_vox = sparse_quantize( 
        coords_pcl, feats_pcl, None, return_index=True,
        return_inverse=True, quantization_size=voxel_size)

    return idx, inverse_idx, coords_vox, feats_vox, labels_pcl, coords_pcl, feats_pcl


def load_file_eval(file_name, voxel_size, attacked_coords, dataset='scannet'):
    """
    Load point clouds for evaluation
    """
    plydata = PlyData.read(file_name+'.ply')
    data = plydata.elements[0].data
    if attacked_coords is not None:
        room_name = file_name.split('/')[-1]
        coords_pcl = np.loadtxt(os.path.join(attacked_coords, room_name + '.txt'))
    else:
        coords_pcl = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
    
    if dataset == 'stanford':
        coords_pcl[:,:2] -= coords_pcl[:,:2].mean(axis=0)
        coords_pcl[:,2] -= coords_pcl[:,2].min(axis=0)
    
    colors_pcl = np.array([data['red'], data['green'],
                       data['blue']], dtype=np.float32).T / 255.
    labels_pcl = np.array(data['label'], dtype=np.int32)

    if dataset == 'scannet':
        feats_pcl = colors_pcl - 0.5
    elif dataset == 'stanford':
        # Normalize feature
        coords_vox = np.floor(coords_pcl / voxel_size)
        coords_vox_mean = coords_vox.mean(0)
        coords_vox_mean[-1] = 0.  # only center x, y!
        coords_vox_norm = coords_vox - coords_vox_mean
        feats_pcl = np.concatenate((colors_pcl - 0.5, coords_vox_norm), 1)

    idx, inverse_idx, coords_vox, feats_vox = sparse_quantize(
        coords_pcl, feats_pcl, None, return_index=True,
        return_inverse=True, quantization_size=voxel_size)

    return inverse_idx, coords_vox, feats_vox, labels_pcl


def generate_input_sparse_tensor(file_name, config, coords_pcl=None, labels_pcl=None, extend=True, dataset='scannet'):
    """
    Obtain sparse tensor for input
    """
    
    batch = [load_file(file_name, config.voxel_size, coords_pcl, labels_pcl, dataset)]
    idx, inverse_idx, coords_vox, feats_vox, labels_pcl, coords_pcl, feats_pcl = list(zip(*batch))
    coords_vox, feats_vox = ME.utils.sparse_collate(coords_vox, feats_vox, None)
    coords_pcl = torch.from_numpy(coords_pcl[0])
    if extend:
        coords_pcl.requires_grad_(True).retain_grad()
        feats_pcl = torch.from_numpy(feats_pcl[0])
        sinput, occupy_conv, valid = add_occupancy(config, inverse_idx, coords_vox[:,1:], coords_pcl, feats_pcl, dataset)
        return idx[0], inverse_idx[0], coords_vox, coords_pcl, sinput, occupy_conv, valid
    else:
        sinput = ME.SparseTensor(feats_vox.float(), coords=coords_vox)
        return idx[0], inverse_idx[0], coords_pcl, sinput, labels_pcl[0]


def generate_input_sparse_tensor_eval(file_name, voxel_size=0.02, attacked_coords=None, dataset='scannet'):
    """
    Obtain sparse tensor for input
    """

    # Create a batch, this process is done in a data loader during training in parallel.
    batch = [load_file_eval(file_name, voxel_size, attacked_coords, dataset)]
    inverse_idx, coordinates_, featrues_, labels = list(zip(*batch))
    coordinates, features = ME.utils.sparse_collate(
        coordinates_, featrues_, None)

    return inverse_idx, coordinates, features.float(), labels[0]


def add_occupancy(config, inverse_idx, coords_vox_noextend, coords_pcl, feats_pcl, dataset='scannet'):
    """
    Obtain occupancy values for input voxelization and sparse convolution
    """

    if dataset == 'stanford':
        # Obtain features with enabled gradient
        feat_size = 6
        feats_pcl[:,3:] = coords_pcl / config.voxel_size
        feats_pcl[:,3:5] -= feats_pcl[:,3:5].mean(0)

    ### Find Valid Voxels ###

    # Find all possible voxels that may be occupied after an attack step.
    coords_vox_all = []
    coords_vox = []
    valid = []
    i = 0
    for dx in [0, -1, 1]:
        for dy in [0, -1, 1]:
            for dz in [0, -1, 1]:
                if config.dynamics_aware == False:
                    if [dx, dy, dz] != [0, 0, 0]:
                        break
                if i == 0:
                    # Add existing occupied voxels
                    valid.append(torch.arange(coords_pcl.shape[0]))     
                    coords_vox_all.append(torch.floor(coords_pcl / config.voxel_size))
                else:
                    # Examine neighbor voxels
                    coords_vox_new = torch.floor((coords_pcl + torch.Tensor([dx, dy, dz]) * config.step) / config.voxel_size)
                    valid.append(torch.where(~((torch.stack(coords_vox_all) - coords_vox_new).abs().sum(-1) == 0).sum(0).bool())[0])
                    coords_vox_all.append(coords_vox_new)
                coords_vox.append(coords_vox_all[i][valid[i]])
                i = i + 1
    coords_vox = torch.cat(coords_vox, dim=0)


    ### Relation Calculation ###

    inverse_idx = torch.Tensor(inverse_idx[0]).long()
    relation_input_list = []
    if config.dynamics_aware:
        relation_conv_list = []
    i = 0
    for dx in [0, -1, 1]:
        for dy in [0, -1, 1]:
            for dz in [0, -1, 1]:
                if config.dynamics_aware == False:
                    if [dx, dy, dz] != [0, 0, 0]:
                        continue

                # Distance
                coords_vox_nei = coords_vox_noextend[inverse_idx][valid[i]] + torch.Tensor([[dx, dy, dz]]) + 0.5
                coords_pcl_valid = coords_pcl[valid[i]]
                dist = torch.abs(coords_vox_nei  - coords_pcl_valid / config.voxel_size)

                # Relation for input voxelization
                relation_input = torch.prod(1/(1+torch.exp(config.lamda_input*(dist-0.5))), dim=-1)
                relation_input_list.append(relation_input)

                # Relation for sparse convolution in network
                if config.dynamics_aware:
                    relation_conv = torch.prod(1/(1+torch.exp(config.lamda_conv*(dist-0.5))), dim=-1)
                    relation_conv_list.append(relation_conv)

                i = i + 1
    relation_input_list = torch.cat(relation_input_list, dim=0)
    relation_conv_list = torch.cat(relation_conv_list, dim=0)


    ### Gathering Operation ###

    # Obtain neighbor mapping in Equations (10) and (18)
    unique_index, inverse_mapping = quantize(coords_vox)

    # Obtain the uniqued voxel coordinates
    coords_vox = coords_vox[unique_index]
    
    # The gathering function in Equation (10)
    occupy_input = 1 - scatter_mul(1-relation_input_list, inverse_mapping)
    if config.dynamics_aware:
        occupy_conv = 1 - scatter_mul(1-relation_conv_list, inverse_mapping)
    

    ### Input Voxelization ###
    
    # Equation (18)
    feats_pcl_list = []
    for i in range(len(valid)):
        feats_pcl_list.append(feats_pcl[valid[i]])
    feats_pcl_list = torch.cat(feats_pcl_list, dim=0)
    mid_result = relation_input_list.unsqueeze(1).repeat(1,feats_pcl.shape[-1]) * feats_pcl_list
    feats_vox_tilde = []
    for i in range(feats_pcl.shape[-1]):
        feats_vox_tilde.append(scatter_sum(mid_result[:,i], inverse_mapping))
    feats_vox_tilde = torch.stack(feats_vox_tilde, dim=1)
    relation_input_sum = scatter_sum(relation_input_list, inverse_mapping)
    if dataset == 'stanford':
        relation_input_sum_ = torch.where(relation_input_sum < 1e-10, torch.ones(relation_input_sum.shape), relation_input_sum)
        relation_input_sum = relation_input_sum_
    feats_vox_tilde = feats_vox_tilde / relation_input_sum.unsqueeze(1)

    if dataset == 'stanford':
        # Use sigmoid to mimic floor operation in coordinate features
        feats_vox_tilde_floor = torch.floor(feats_vox_tilde[:,3:] - 0.5)
        feats_vox_tilde_rem = 1 / (1 + torch.exp(-config.lamda_floor * (feats_vox_tilde[:,3:] - feats_vox_tilde_floor)))
        feats_vox_tilde[:,3:] = feats_vox_tilde_floor + feats_vox_tilde_rem

    # Equation (19)
    feats_vox = occupy_input.unsqueeze(1).repeat(1,feats_pcl.shape[-1]) * feats_vox_tilde

    # Build input sparse tensor
    sparse_tensor = ME.SparseTensor(feats_vox, coords=torch.cat([torch.zeros(coords_vox.shape[0],1), coords_vox], dim=-1).int())
    sparse_tensor._F = feats_vox


    # Return input sparse tensor, occupancy values for sparse convolution in network, and valid index
    if config.dynamics_aware:
        return sparse_tensor, occupy_conv.unsqueeze(1), valid
    else:
        return sparse_tensor, None, valid


def get_point_output(config, soutput, inverse_idx, coords_vox_noextend, coords_pcl, valid):
    """
    Obtain occupancy values for output devoxelization
    """

    ### Output Devoxelization ###

    # Note:
    #   Equation (20) is applied by not multiplying occupancy value on the final layer output of the network.
    #   So in devoxelization we only need to apply equation (21).
    
    outputs_pcl = torch.zeros(coords_pcl.shape[0], soutput.F.shape[1]).to(soutput.device)
    i = 0
    for dx in [0, -1, 1]:
        for dy in [0, -1, 1]:
            for dz in [0, -1, 1]:
                if config.dynamics_aware == False:
                    if [dx, dy, dz] != [0, 0, 0]:
                        continue
                
                # Distance
                coords_vox_nei = coords_vox_noextend[:,1:][inverse_idx][valid[i]] + torch.Tensor([[dx, dy, dz]]) + 0.5
                coords_pcl_valid = coords_pcl[valid[i]]
                dist = torch.abs(coords_vox_nei  - coords_pcl_valid / config.voxel_size)

                # Relation for output devoxelization
                relation_output = torch.prod(1/(1+torch.exp(config.lamda_output*(dist-0.5))), dim=-1)
                relation_output = relation_output.to(soutput.device)
                
                # We ignore the denominator in Equation (21) for simplicity
                try:
                    outputs_vox = soutput.features_at_coords(coords_vox_noextend + torch.IntTensor([[0, dx, dy, dz]]))[0][inverse_idx][valid[i]]
                except:
                    i = i + 1
                    continue
                else:
                    # Equation (21)
                    outputs = relation_output.unsqueeze(1) * outputs_vox
                    outputs_pcl[valid[i]] += outputs
                    i = i + 1
    return outputs_pcl


def save_prediction(config, save_path, room_name, preds, probs=None, dataset='scannet'):
    """
    Save network prediction
    """
    if dataset == 'scannet':
        VALID_CLASS_IDS = SCANNET_VALID_CLASS_IDS
    elif dataset == 'stanford':
        VALID_CLASS_IDS = STANFORD_VALID_CLASS_IDS

    if config.save_preds:
        preds = np.array(VALID_CLASS_IDS, dtype=np.int32)[preds]
        if not os.path.exists(os.path.join(save_path, 'pred')):
            os.makedirs(os.path.join(save_path, 'pred'))
        np.savetxt(os.path.join(save_path, 'pred', room_name+'.txt'), preds, fmt='%d')

    if config.save_probs:
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
    np.savetxt(os.path.join(save_path, 'coord', room_name+'.txt'), coords_pcl, fmt='%.6f')
    return


def visualize(config, room_name, coords_pcl, labels_pcl, save_path, remark=None, refine=False):
    """
    Function for visualization
    """
    if refine:
        coords_pcl = torch.Tensor(coords_pcl)
        coords_pcl_ = config.voxel_size * torch.floor(coords_pcl/config.voxel_size) + config.voxel_size * torch.rand(coords_pcl.shape)
        coords_pcl_ = torch.where(coords_pcl_ < (coords_pcl0 - config.budget), coords_pcl0 - config.budget, coords_pcl_)
        coords_pcl_ = torch.where(coords_pcl_ > (coords_pcl0 + config.budget), coords_pcl0 + config.budget, coords_pcl_)
        coords_pcl = torch.where(torch.floor(coords_pcl_/config.voxel_size) != torch.floor(coords_pcl/config.voxel_size), coords_pcl, coords_pcl_)
        coords_pcl = coords_pcl.numpy()

    plydata = PlyData.read(os.path.join(config.data_path, room_name + '.ply'))
    if remark == 'gt':
        rgb = []
        for l in labels_pcl:
            if l == 255:
                rgb.append(list(colors[0]))
            else:
                rgb.append(list(colors[l+1]))
        rgb = np.array(rgb)
    elif remark == 'noattack':
        rgb = np.array([list(colors[l+1]) for l in labels_pcl])
    else:
        if remark == 'attack':
            plydata.elements[0].data['x'] = coords_pcl[:, 0]
            plydata.elements[0].data['y'] = coords_pcl[:, 1]
            plydata.elements[0].data['z'] = coords_pcl[:, 2]
        rgb = np.array([list(colors[l+1]) for l in labels_pcl])

    plydata.elements[0].data['red'] = rgb[:, 0]
    plydata.elements[0].data['green'] = rgb[:, 1]
    plydata.elements[0].data['blue'] = rgb[:, 2]

    if remark is not None:
        save_path = os.path.join(save_path, 'visual', room_name)
    else:
        save_path = os.path.join(save_path, 'visual')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if remark is not None:
        plydata.write(os.path.join(save_path, room_name + '.' + remark + '.ply'))
    else:
        plydata.write(os.path.join(save_path, room_name + '.ply'))



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


def quantize(coords):
    r"""Returns a unique index map and an inverse index map.

    Args:
        :attr:`coords` (:attr:`numpy.ndarray` or :attr:`torch.Tensor`): a
        matrix of size :math:`N \times D` where :math:`N` is the number of
        points in the :math:`D` dimensional space.

    Returns:
        :attr:`unique_map` (:attr:`numpy.ndarray` or :attr:`torch.Tensor`): a
        list of indices that defines unique coordinates.
        :attr:`coords[unique_map]` is the unique coordinates.

        :attr:`inverse_map` (:attr:`numpy.ndarray` or :attr:`torch.Tensor`): a
        list of indices that defines the inverse map that recovers the original
        coordinates.  :attr:`coords[unique_map[inverse_map]] == coords`

    Example::

       >>> unique_map, inverse_map = quantize(coords)
       >>> unique_coords = coords[unique_map]
       >>> print(unique_coords[inverse_map] == coords)  # True, ..., True
       >>> print(coords[unique_map[inverse_map]] == coords)  # True, ..., True

    """
    assert isinstance(coords, np.ndarray) or isinstance(coords, torch.Tensor), \
        "Invalid coords type"
    if isinstance(coords, np.ndarray):
        assert coords.dtype == np.int32, f"Invalid coords type {coords.dtype} != np.int32"
        return MEB.quantize_np(coords.astype(np.int32))
    else:
        # Type check done inside
        return MEB.quantize_th(coords.int())


def quantize_label(coords, labels, ignore_label):
    assert isinstance(coords, np.ndarray) or isinstance(coords, torch.Tensor), \
        "Invalid coords type"
    if isinstance(coords, np.ndarray):
        assert isinstance(labels, np.ndarray)
        assert coords.dtype == np.int32, f"Invalid coords type {coords.dtype} != np.int32"
        assert labels.dtype == np.int32, f"Invalid label type {labels.dtype} != np.int32"
        return MEB.quantize_label_np(coords, labels, ignore_label)
    else:
        assert isinstance(labels, torch.Tensor)
        # Type check done inside
        return MEB.quantize_label_th(coords, labels.int(), ignore_label)


def sparse_quantize(coords,
                    feats=None,
                    labels=None,
                    ignore_label=-100,
                    return_index=False,
                    return_inverse=False,
                    quantization_size=None):
    r"""Given coordinates, and features (optionally labels), the function
    generates quantized (voxelized) coordinates.

    Args:
        :attr:`coords` (:attr:`numpy.ndarray` or :attr:`torch.Tensor`): a
        matrix of size :math:`N \times D` where :math:`N` is the number of
        points in the :math:`D` dimensional space.

        :attr:`feats` (:attr:`numpy.ndarray` or :attr:`torch.Tensor`, optional): a
        matrix of size :math:`N \times D_F` where :math:`N` is the number of
        points and :math:`D_F` is the dimension of the features. Must have the
        same container as `coords` (i.e. if `coords` is a torch.Tensor, `feats`
        must also be a torch.Tensor).

        :attr:`labels` (:attr:`numpy.ndarray` or :attr:`torch.IntTensor`,
        optional): integer labels associated to eah coordinates.  Must have the
        same container as `coords` (i.e. if `coords` is a torch.Tensor,
        `labels` must also be a torch.Tensor). For classification where a set
        of points are mapped to one label, do not feed the labels.

        :attr:`ignore_label` (:attr:`int`, optional): the int value of the
        IGNORE LABEL.
        :attr:`torch.nn.CrossEntropyLoss(ignore_index=ignore_label)`

        :attr:`return_index` (:attr:`bool`, optional): set True if you want the
        indices of the quantized coordinates. False by default.

        :attr:`return_inverse` (:attr:`bool`, optional): set True if you want
        the indices that can recover the discretized original coordinates.
        False by default. `return_index` must be True when `return_reverse` is True.

        Example::

           >>> unique_map, inverse_map = sparse_quantize(discrete_coords, return_index=True, return_inverse=True)
           >>> unique_coords = discrete_coords[unique_map]
           >>> print(unique_coords[inverse_map] == discrete_coords)  # True

        :attr:`quantization_size` (:attr:`float`, :attr:`list`, or
        :attr:`numpy.ndarray`, optional): the length of the each side of the
        hyperrectangle of of the grid cell.

     Example::

        >>> # Segmentation
        >>> criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        >>> coords, feats, labels = MinkowskiEngine.utils.sparse_quantize(
        >>>     coords, feats, labels, ignore_label=-100, quantization_size=0.1)
        >>> output = net(MinkowskiEngine.SparseTensor(feats, coords))
        >>> loss = criterion(output.F, labels.long())
        >>>
        >>> # Classification
        >>> criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        >>> coords, feats = MinkowskiEngine.utils.sparse_quantize(coords, feats)
        >>> output = net(MinkowskiEngine.SparseTensor(feats, coords))
        >>> loss = criterion(output.F, labels.long())


    """
    assert isinstance(coords, np.ndarray) or isinstance(coords, torch.Tensor), \
        'Coords must be either np.array or torch.Tensor.'

    use_label = labels is not None
    use_feat = feats is not None

    assert coords.ndim == 2, \
        "The coordinates must be a 2D matrix. The shape of the input is " + \
        str(coords.shape)

    if return_inverse:
        assert return_index, "return_reverse must be set with return_index"

    if use_feat:
        assert feats.ndim == 2
        assert coords.shape[0] == feats.shape[0]

    if use_label:
        assert coords.shape[0] == len(labels)

    dimension = coords.shape[1]
    # Quantize the coordinates
    if quantization_size is not None:
        if isinstance(quantization_size, (Sequence, np.ndarray, torch.Tensor)):
            assert len(
                quantization_size
            ) == dimension, "Quantization size and coordinates size mismatch."
            if isinstance(coords, np.ndarray):
                quantization_size = np.array([i for i in quantization_size])
                discrete_coords = np.floor(coords / quantization_size)
            else:
                quantization_size = torch.Tensor(
                    [i for i in quantization_size])
                discrete_coords = (coords / quantization_size).floor()

        elif np.isscalar(quantization_size):  # Assume that it is a scalar

            if quantization_size == 1:
                discrete_coords = coords
            else:
                discrete_coords = np.floor(coords / quantization_size)
        else:
            raise ValueError('Not supported type for quantization_size.')
    else:
        discrete_coords = coords

    discrete_coords = np.floor(discrete_coords)
    if isinstance(coords, np.ndarray):
        discrete_coords = discrete_coords.astype(np.int32)
    else:
        discrete_coords = discrete_coords.int()

    # Return values accordingly
    if use_label:
        mapping, colabels = quantize_label(discrete_coords, labels,
                                           ignore_label)

        if return_index:
            return discrete_coords[mapping], feats[mapping], colabels, mapping
        else:
            if use_feat:
                return discrete_coords[mapping], feats[mapping], colabels
            else:
                return discrete_coords[mapping], colabels

    else:
        unique_map, inverse_map = quantize(discrete_coords)
        if return_index:
            if return_inverse:
                return unique_map, inverse_map, \
                    discrete_coords[unique_map], feats[unique_map]
            else:
                return unique_map, \
                    discrete_coords[unique_map], feats[unique_map]
        else:
            if use_feat:
                return discrete_coords[unique_map], feats[unique_map]
            else:
                return discrete_coords[unique_map]


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = 255
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target