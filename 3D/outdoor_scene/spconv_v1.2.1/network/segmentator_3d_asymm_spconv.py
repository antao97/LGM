# network/segmentator_3d_asymm_spconv.py
# -*- coding:utf-8 -*-
# author: Xinge
# @file: segmentator_3d_asymm_spconv.py
# 
# modified by: An Tao, Pengliang Ji

from pickle import NONE
import numpy as np
import random
import spconv
import spconv.functional as Fsp
from spconv import ops
import torch
from torch import nn


def change_occupy(occupy, indicePair, out_shape):
    occupy_new = torch.ones(out_shape).unsqueeze(1).to(occupy.device)
    for i in range(indicePair.shape[1]):
        in_pair = indicePair[0,i][indicePair[0,i]!=-1].long()
        out_pair = indicePair[1,i][indicePair[1,i]!=-1].long()
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
            shortcut.features = shortcut.features * occupy
        shortcut.features = self.act1(shortcut.features)
        shortcut.features = self.bn0(shortcut.features)
        if occupy is not None:
            shortcut.features = shortcut.features * occupy

        shortcut = self.conv1_2(shortcut)
        if occupy is not None:
            shortcut.features = shortcut.features * occupy
        shortcut.features = self.act1_2(shortcut.features)
        shortcut.features = self.bn0_2(shortcut.features)
        if occupy is not None:
            shortcut.features = shortcut.features * occupy

        resA = self.conv2(x)
        if occupy is not None:
            resA.features = resA.features * occupy
        resA.features = self.act2(resA.features)
        resA.features = self.bn1(resA.features)
        if occupy is not None:
            resA.features = resA.features * occupy

        resA = self.conv3(resA)
        if occupy is not None:
            resA.features = resA.features * occupy
        resA.features = self.act3(resA.features)
        resA.features = self.bn2(resA.features)
        if occupy is not None:
            resA.features = resA.features * occupy
        resA.features = resA.features + shortcut.features

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
            shortcut.features = shortcut.features * occupy
        shortcut.features = self.act1(shortcut.features)
        shortcut.features = self.bn0(shortcut.features)
        if occupy is not None:
            shortcut.features = shortcut.features * occupy

        shortcut = self.conv1_2(shortcut)
        if occupy is not None:
            shortcut.features = shortcut.features * occupy
        shortcut.features = self.act1_2(shortcut.features)
        shortcut.features = self.bn0_2(shortcut.features)
        if occupy is not None:
            shortcut.features = shortcut.features * occupy

        resA = self.conv2(x)
        if occupy is not None:
            resA.features = resA.features * occupy
        resA.features = self.act2(resA.features)
        resA.features = self.bn1(resA.features)
        if occupy is not None:
            resA.features = resA.features * occupy

        resA = self.conv3(resA)
        if occupy is not None:
            resA.features = resA.features * occupy
        resA.features = self.act3(resA.features)
        resA.features = self.bn2(resA.features)
        if occupy is not None:
            resA.features = resA.features * occupy
        resA.features = resA.features + shortcut.features

        if self.pooling:
            resB = self.pool(resA)
            if occupy is not None:
                occypy1 = change_occupy(occupy, resB.indice_dict[self.indice_key][2], resB.indices.shape[0])
                resB.features = resB.features * occypy1
                return resB, resA, occypy1
            return resB, resA
        else:
            return resA


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), indice_key=None, up_key=None):
        super(UpBlock, self).__init__()
        # self.drop_out = drop_out
        self.trans_dilao = conv3x3(in_filters, out_filters, indice_key=indice_key + "new_up")
        self.trans_act = nn.LeakyReLU()
        self.trans_bn = nn.BatchNorm1d(out_filters)

        self.conv1 = conv1x3(out_filters, out_filters, indice_key=indice_key)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv3x1(out_filters, out_filters, indice_key=indice_key)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x3(out_filters, out_filters, indice_key=indice_key)
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
            upA.features = upA.features * occupy
        upA.features = self.trans_act(upA.features)
        upA.features = self.trans_bn(upA.features)
        if occupy is not None:
            upA.features = upA.features * occupy

        upA = self.up_subm(upA)
        if occupy_up is not None:
            upA.features = upA.features * occupy_up

        upA.features = upA.features + skip.features

        upE = self.conv1(upA)
        if occupy_up is not None:
            upE.features = upE.features * occupy_up
        upE.features = self.act1(upE.features)
        upE.features = self.bn1(upE.features)
        if occupy_up is not None:
            upE.features = upE.features * occupy_up

        upE = self.conv2(upE)
        if occupy_up is not None:
            upE.features = upE.features * occupy_up
        upE.features = self.act2(upE.features)
        upE.features = self.bn2(upE.features)
        if occupy_up is not None:
            upE.features = upE.features * occupy_up

        upE = self.conv3(upE)
        if occupy_up is not None:
            upE.features = upE.features * occupy_up
        upE.features = self.act3(upE.features)
        upE.features = self.bn3(upE.features)
        if occupy_up is not None:
            upE.features = upE.features * occupy_up

        return upE


class ReconBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ReconBlock, self).__init__()
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
            shortcut.features = shortcut.features * occupy
        shortcut.features = self.bn0(shortcut.features)
        if occupy is not None:
            shortcut.features = shortcut.features * occupy
        shortcut.features = self.act1(shortcut.features)

        shortcut2 = self.conv1_2(x)
        if occupy is not None:
            shortcut2.features = shortcut2.features * occupy
        shortcut2.features = self.bn0_2(shortcut2.features)
        if occupy is not None:
            shortcut2.features = shortcut2.features * occupy
        shortcut2.features = self.act1_2(shortcut2.features)

        shortcut3 = self.conv1_3(x)
        if occupy is not None:
            shortcut3.features = shortcut3.features * occupy
        shortcut3.features = self.bn0_3(shortcut3.features)
        if occupy is not None:
            shortcut3.features = shortcut3.features * occupy
        shortcut3.features = self.act1_3(shortcut3.features)
        shortcut.features = shortcut.features + shortcut2.features + shortcut3.features

        shortcut.features = shortcut.features * x.features

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

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c, down4b = self.resBlock5(down3c)

        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        up0e.features = torch.cat((up0e.features, up1e.features), 1)

        logits = self.logits(up0e)
        y = logits.dense()
        return y


class New_Asymm_3d_spconv(Asymm_3d_spconv):
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

        up0e.features = torch.cat((up0e.features, up1e.features), 1)

        logits = self.logits(up0e)
        logits.features = logits.features * occupy
        y = logits.dense()
        return y