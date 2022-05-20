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

import torch
import torch.nn.functional as F
from torch_scatter import scatter_mul
import MinkowskiEngine as ME
import utils
from SparseTensor import _get_coords_key


def conv_nosample(conv, x, occupy, real_num, stride=1, mul_occupy_after=True):
    """
    Add occupancy value for sparse convolution that does not change the stride for input data
    """

    if occupy.ndim < 2:
        occupy = occupy.unsqueeze(1)

    if conv.kernel.ndim < 3:
        conv.kernel.data = conv.kernel.data.unsqueeze(0)

    conv.kernel.requires_grad_(False)
    if conv.bias is not None:
        conv.bias.requires_grad_(False)

    out0_coords_key = _get_coords_key(x, x.C[:real_num])
    out0_feat = conv(x, out0_coords_key).F

    if x.C[real_num:].shape[0] > 0:
        out1_coords_key = _get_coords_key(x, x.C[real_num:])
        out1_feat = conv(x, out1_coords_key).F
        out_feat = torch.cat([out0_feat, out1_feat], dim=0)
    else:
        out_feat = out0_feat
    out = ME.SparseTensor(out_feat, coords=x.C, tensor_stride=stride)

    if mul_occupy_after:
        out._F = out._F * occupy
    return out


def conv_downsample(conv, x, occupy, real_num, out_stride=1, mul_occupy_after=True):
    """
    Add occupancy value for sparse convolution that increases the stride for input data
    """

    if occupy.ndim < 2:
        occupy = occupy.unsqueeze(1)

    if conv.kernel.ndim < 3:
        conv.kernel.data = conv.kernel.data.unsqueeze(0)

    conv.kernel.requires_grad_(False)
    if conv.bias is not None:
        conv.bias.requires_grad_(False)

    unique_map, inverse_map = utils.quantize(torch.floor(x.C.float() / out_stride).int() * out_stride)
    out_coords = (torch.floor(x.C.float() / out_stride).int() * out_stride)[unique_map]
    occupy_new = 1 - scatter_mul(1-occupy.view(-1), inverse_map.to(occupy.device)).unsqueeze(1)

    real_num_new = utils.quantize(torch.floor(x.C.float()[:real_num] / out_stride).int() * out_stride)[0].shape[0]
    
    out0_coords_key = _get_coords_key(x, out_coords[:real_num_new], tensor_stride=out_stride)
    out0_feat = conv(x, out0_coords_key).F

    if out_coords[real_num_new:].shape[0] > 0:
        out1_coords_key = _get_coords_key(x, out_coords[real_num_new:], tensor_stride=out_stride)
        out1_feat = conv(x, out1_coords_key).F
        out_feat = torch.cat([out0_feat, out1_feat], dim=0)
    else:
        out_feat = out0_feat
    out = ME.SparseTensor(out_feat, coords=out_coords, tensor_stride=out_stride)

    if mul_occupy_after:
        out._F = out._F * occupy_new
    return out, occupy_new, real_num_new


def conv_upample(conv, x, occupy, real_num, out_coords, out_occupy, out_stride=1, mul_occupy_after=True):
    """
    Add occupancy value for sparse convolution that decreases the stride for input data
    """

    if occupy.ndim < 2:
        occupy = occupy.unsqueeze(1)

    if conv.kernel.ndim < 3:
        conv.kernel.data = conv.kernel.data.unsqueeze(0)

    conv.kernel.requires_grad_(False)
    if conv.bias is not None:
        conv.bias.requires_grad_(False)
    
    x0 = ME.SparseTensor(x.F[:real_num], coords=x.C[:real_num], tensor_stride=x.tensor_stride)
    out0_coords_key = _get_coords_key(x0, out_coords, tensor_stride=out_stride)
    out0_feat = conv(x0, out0_coords_key).F

    if x.C[real_num:].shape[0] > 0:
        x1 = ME.SparseTensor(x.F[real_num:], coords=x.C[real_num:], tensor_stride=x.tensor_stride)
        out1_coords_key = _get_coords_key(x1, out_coords, tensor_stride=out_stride)
        out1_feat = conv(x1, out1_coords_key).F
        out_feat = out0_feat + out1_feat
    else:
        out_feat = out0_feat
    out = ME.SparseTensor(out_feat, coords=out_coords, tensor_stride=out_stride)

    if mul_occupy_after:
        out._F = out._F * out_occupy
    return out


def fn_forward(fn, x, occupy, mul_occupy_after=True):
    """
    Add occupancy value for normal forward function
    """

    if occupy.ndim < 2:
        occupy = occupy.unsqueeze(1)

    fn.bn.requires_grad_(False)

    out = fn(x)
    if mul_occupy_after:
        out._F = out._F * occupy
    return out

