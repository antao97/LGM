# -*- coding:utf-8 -*-
# author: Xinge
# @file: segmentator_3d_asymm_spconv.py
# 
# modified by: An Tao, Pengliang Ji

from pickle import NONE
import numpy as np
import random
# import spconv
import spconv.pytorch as spconv
# import spconv.functional as Fsp
from spconv.pytorch import functional as Fsp
# from spconv import ops
from spconv.pytorch import ops
import torch
from torch import nn


def change_occupy(occupy, pair, out_shape):
    occupy_new = torch.ones(out_shape).unsqueeze(1).to(occupy.device)
    for i in range(pair.pair_fwd.shape[0]):
        out_pair = (pair.pair_fwd[i] != -1)
        in_pair = pair.pair_fwd[i][out_pair].long()
        occupy_new[out_pair] *= 1 - occupy[in_pair]
    occupy_new = 1 - occupy_new
    return occupy_new


def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


def conv1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride,
                             padding=(0, 1, 1), bias=False, indice_key=indice_key)


def conv1x1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride,
                             padding=(0, 0, 1), bias=False, indice_key=indice_key)


def conv1x3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 1), stride=stride,
                             padding=(0, 1, 0), bias=False, indice_key=indice_key)


def conv3x1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride,
                             padding=(1, 0, 0), bias=False, indice_key=indice_key)


def conv3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 3), stride=stride,
                             padding=(1, 0, 1), bias=False, indice_key=indice_key)


def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ResContextBlock, self).__init__()
        self.conv1 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.LeakyReLU()

        self.conv1_2 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.LeakyReLU()

        self.conv2 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, occupy=None):
        shortcut = self.conv1(x)
        if occupy is not None:
            shortcut = shortcut.replace_feature(shortcut.features * occupy)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))
        if occupy is not None:
            shortcut = shortcut.replace_feature(shortcut.features * occupy)

        shortcut.indice_dict["prebef"].ksize = [3,1,3]
        shortcut = self.conv1_2(shortcut)
        if occupy is not None:
            shortcut = shortcut.replace_feature(shortcut.features * occupy)
        shortcut = shortcut.replace_feature(self.act1_2(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_2(shortcut.features))
        if occupy is not None:
            shortcut = shortcut.replace_feature(shortcut.features * occupy)

        x.indice_dict["prebef"] = shortcut.indice_dict["prebef"]
        resA = self.conv2(x)
        if occupy is not None:
            resA = resA.replace_feature(resA.features * occupy)
        resA = resA.replace_feature(self.act2(resA.features))
        resA = resA.replace_feature(self.bn1(resA.features))
        if occupy is not None:
            resA = resA.replace_feature(resA.features * occupy)

        resA.indice_dict["prebef"].ksize = [1,3,3]
        resA = self.conv3(resA)
        if occupy is not None:
            resA = resA.replace_feature(resA.features * occupy)
        resA = resA.replace_feature(self.act3(resA.features))
        resA = resA.replace_feature(self.bn2(resA.features))
        if occupy is not None:
            resA = resA.replace_feature(resA.features * occupy)
        resA = resA.replace_feature(resA.features + shortcut.features)

        return resA


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3, 3), stride=1,
                 pooling=True, drop_out=True, height_pooling=False, indice_key=None):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.height_pooling = height_pooling
        self.indice_key = indice_key

        self.conv1 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.act1 = nn.LeakyReLU()
        self.bn0 = nn.BatchNorm1d(out_filters)

        self.conv1_2 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act1_2 = nn.LeakyReLU()
        self.bn0_2 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        if pooling:
            if height_pooling:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=2,
                                                padding=1, indice_key=indice_key, bias=False)
            else:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=(2, 2, 1),
                                                padding=1, indice_key=indice_key, bias=False)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, occupy=None):
        shortcut = self.conv1(x)
        if occupy is not None:
            shortcut = shortcut.replace_feature(shortcut.features * occupy)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))
        if occupy is not None:
            shortcut = shortcut.replace_feature(shortcut.features * occupy)

        shortcut.indice_dict[self.indice_key + "bef"].ksize = [1,3,3]
        shortcut = self.conv1_2(shortcut)
        if occupy is not None:
            shortcut = shortcut.replace_feature(shortcut.features * occupy)
        shortcut = shortcut.replace_feature(self.act1_2(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_2(shortcut.features))
        if occupy is not None:
            shortcut = shortcut.replace_feature(shortcut.features * occupy)

        x.indice_dict[self.indice_key + "bef"] = shortcut.indice_dict[self.indice_key + "bef"]
        resA = self.conv2(x)
        if occupy is not None:
            resA = resA.replace_feature(resA.features * occupy)
        resA = resA.replace_feature(self.act2(resA.features))
        resA = resA.replace_feature(self.bn1(resA.features))
        if occupy is not None:
            resA = resA.replace_feature(resA.features * occupy)

        resA.indice_dict[self.indice_key + "bef"].ksize = [3,1,3]
        resA = self.conv3(resA)
        if occupy is not None:
            resA = resA.replace_feature(resA.features * occupy)
        resA = resA.replace_feature(self.act3(resA.features))
        resA = resA.replace_feature(self.bn2(resA.features))
        if occupy is not None:
            resA = resA.replace_feature(resA.features * occupy)
        resA = resA.replace_feature(resA.features + shortcut.features)

        if self.pooling:
            resB = self.pool(resA)
            if occupy is not None:
                occupy_new = change_occupy(occupy, resB.indice_dict[self.indice_key], resB.indices.shape[0])
                resB = resB.replace_feature(resB.features * occupy_new)
            else:
                occupy_new = None
            return resB, resA, occupy_new
        else:
            return resA


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), indice_key=None, up_key=None):
        super(UpBlock, self).__init__()
        # self.drop_out = drop_out
        self.indice_key = indice_key
        self.trans_dilao = conv3x3(in_filters, out_filters, indice_key=indice_key + "new_up")
        self.trans_act = nn.LeakyReLU()
        self.trans_bn = nn.BatchNorm1d(out_filters)

        self.conv1 = conv1x3(out_filters, out_filters, indice_key=indice_key)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv3x1(out_filters, out_filters, indice_key=indice_key)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        # self.conv3 = conv3x3(out_filters, out_filters, indice_key=indice_key)
        self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(out_filters)
        # self.dropout3 = nn.Dropout3d(p=dropout_rate)

        self.up_subm = spconv.SparseInverseConv3d(out_filters, out_filters, kernel_size=3, indice_key=up_key,
                                                  bias=False)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, skip, occupy=None, occupy_up=None):
        upA = self.trans_dilao(x)
        if occupy is not None:
            upA = upA.replace_feature(upA.features * occupy)
        upA = upA.replace_feature(self.trans_act(upA.features))
        upA = upA.replace_feature(self.trans_bn(upA.features))
        if occupy is not None:
            upA = upA.replace_feature(upA.features * occupy)

        upA = self.up_subm(upA)
        if occupy_up is not None:
            upA = upA.replace_feature(upA.features * occupy_up)

        upA = upA.replace_feature(upA.features + skip.features)

        upE = self.conv1(upA)
        if occupy_up is not None:
            upE = upE.replace_feature(upE.features * occupy_up)
        upE = upE.replace_feature(self.act1(upE.features))
        upE = upE.replace_feature(self.bn1(upE.features))
        if occupy_up is not None:
            upE = upE.replace_feature(upE.features * occupy_up)

        upE.indice_dict[self.indice_key].ksize = [3,1,3]
        upE = self.conv2(upE)
        if occupy_up is not None:
            upE = upE.replace_feature(upE.features * occupy_up)
        upE = upE.replace_feature(self.act2(upE.features))
        upE = upE.replace_feature(self.bn2(upE.features))
        if occupy_up is not None:
            upE = upE.replace_feature(upE.features * occupy_up)

        upE.indice_dict[self.indice_key].ksize = [1,3,3]
        upE = self.conv3(upE)
        if occupy_up is not None:
            upE = upE.replace_feature(upE.features * occupy_up)
        upE = upE.replace_feature(self.act3(upE.features))
        upE = upE.replace_feature(self.bn3(upE.features))
        if occupy_up is not None:
            upE = upE.replace_feature(upE.features * occupy_up)

        return upE


class ReconBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ReconBlock, self).__init__()
        self.indice_key = indice_key
        self.conv1 = conv3x1x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.Sigmoid()

        self.conv1_2 = conv1x3x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.Sigmoid()

        self.conv1_3 = conv1x1x3(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0_3 = nn.BatchNorm1d(out_filters)
        self.act1_3 = nn.Sigmoid()

    def forward(self, x, occupy=None):
        shortcut = self.conv1(x)
        if occupy is not None:
            shortcut = shortcut.replace_feature(shortcut.features * occupy)
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))
        if occupy is not None:
            shortcut = shortcut.replace_feature(shortcut.features * occupy)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))

        x.indice_dict[self.indice_key + "bef"] = shortcut.indice_dict[self.indice_key + "bef"]
        x.indice_dict[self.indice_key + "bef"].ksize = [1,3,1]
        shortcut2 = self.conv1_2(x)
        if occupy is not None:
            shortcut2 = shortcut2.replace_feature(shortcut2.features * occupy)
        shortcut2 = shortcut2.replace_feature(self.bn0_2(shortcut2.features))
        if occupy is not None:
            shortcut2 = shortcut2.replace_feature(shortcut2.features * occupy)
        shortcut2 = shortcut2.replace_feature(self.act1_2(shortcut2.features))

        x.indice_dict[self.indice_key + "bef"] = shortcut.indice_dict[self.indice_key + "bef"]
        x.indice_dict[self.indice_key + "bef"].ksize = [1,1,3]
        shortcut3 = self.conv1_3(x)
        if occupy is not None:
            shortcut3 = shortcut3.replace_feature(shortcut3.features * occupy)
        shortcut3 = shortcut3.replace_feature(self.bn0_3(shortcut3.features))
        if occupy is not None:
            shortcut3 = shortcut3.replace_feature(shortcut3.features * occupy)
        shortcut3 = shortcut3.replace_feature(self.act1_3(shortcut3.features))
        shortcut = shortcut3.replace_feature(shortcut.features + shortcut2.features + shortcut3.features)

        shortcut = shortcut.replace_feature(shortcut.features * x.features)

        return shortcut


class Asymm_3d_spconv(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16):
        super(Asymm_3d_spconv, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)

    def forward(self, voxel_features, coors, batch_size, occupy=None):

        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret, occupy)
        down1c, down1b, occupy1 = self.resBlock2(ret, occupy)
        down2c, down2b, occupy2 = self.resBlock3(down1c, occupy1)
        down3c, down3b, occupy3 = self.resBlock4(down2c, occupy2)
        down4c, down4b, occupy4 = self.resBlock5(down3c, occupy3)

        up4e = self.upBlock0(down4c, down4b, occupy4, occupy3)
        up3e = self.upBlock1(up4e, down3b, occupy3, occupy2)
        up2e = self.upBlock2(up3e, down2b, occupy2, occupy1)
        up1e = self.upBlock3(up2e, down1b, occupy1, occupy)
        up0e = self.ReconNet(up1e, occupy)

        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

        logits = self.logits(up0e)
        if occupy is not None:
            logits = logits.replace_feature(logits.features * occupy)
        y = logits.dense()
        return y