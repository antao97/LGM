# -*- coding:utf-8 -*-
# author: Xinge
# 
# modified by: An Tao, Pengliang Ji

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_scatter
import numba as nb
# import spconv
import spconv.pytorch as spconv
import multiprocessing


class cylinder_fea(nn.Module):

    def __init__(self, grid_size, fea_dim=3,
                 out_pt_fea_dim=64, max_pt_per_encode=64, fea_compre=None):
        super(cylinder_fea, self).__init__()

        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(fea_dim),
            nn.Linear(fea_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, out_pt_fea_dim)
        )

        self.max_pt = max_pt_per_encode
        self.fea_compre = fea_compre
        self.grid_size = grid_size
        kernel_size = 3
        self.local_pool_op = torch.nn.MaxPool2d(kernel_size, stride=1,
                                                padding=(kernel_size - 1) // 2,
                                                dilation=1)
        self.pool_dim = out_pt_fea_dim

        # point feature compression
        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(self.pool_dim, self.fea_compre),
                nn.ReLU())
            self.pt_fea_dim = self.fea_compre
        else:
            self.pt_fea_dim = self.pool_dim

    def forward(self, pt_fea, xy_ind, occupy):
        cat_pt_ind = []
        for i_batch in range(len(xy_ind)):
            cat_pt_ind.append(F.pad(xy_ind[i_batch], (1, 0), 'constant', value=i_batch))
        
        cat_pt_fea = torch.cat(pt_fea, dim=0)
        cat_pt_ind = torch.cat(cat_pt_ind, dim=0)
       
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind, return_inverse=True, return_counts=True, dim=0)
        unq = unq.type(torch.int64)

        processed_cat_pt_fea = self.PPmodel[0](cat_pt_fea)
        if occupy is not None:
            processed_cat_pt_fea = processed_cat_pt_fea * occupy
        processed_cat_pt_fea = self.PPmodel[1](processed_cat_pt_fea)
        if occupy is not None:
            processed_cat_pt_fea = processed_cat_pt_fea * occupy
        processed_cat_pt_fea = self.PPmodel[2](processed_cat_pt_fea)
        if occupy is not None:
            processed_cat_pt_fea = processed_cat_pt_fea * occupy
        processed_cat_pt_fea = self.PPmodel[3](processed_cat_pt_fea)
        processed_cat_pt_fea = self.PPmodel[4](processed_cat_pt_fea)
        if occupy is not None:
            processed_cat_pt_fea = processed_cat_pt_fea * occupy
        processed_cat_pt_fea = self.PPmodel[5](processed_cat_pt_fea)
        if occupy is not None:
            processed_cat_pt_fea = processed_cat_pt_fea * occupy
        processed_cat_pt_fea = self.PPmodel[6](processed_cat_pt_fea)
        processed_cat_pt_fea = self.PPmodel[7](processed_cat_pt_fea)
        if occupy is not None:
            processed_cat_pt_fea = processed_cat_pt_fea * occupy
        processed_cat_pt_fea = self.PPmodel[8](processed_cat_pt_fea)
        if occupy is not None:
            processed_cat_pt_fea = processed_cat_pt_fea * occupy
        processed_cat_pt_fea = self.PPmodel[9](processed_cat_pt_fea)
        processed_cat_pt_fea = self.PPmodel[10](processed_cat_pt_fea)
        if occupy is not None:
            processed_cat_pt_fea = processed_cat_pt_fea * occupy

        pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0]
        if occupy is not None:
            unq_idx = torch_scatter.scatter_max(torch.arange(unq_inv.shape[0]), unq_inv.cpu())[0]
            pooled_occupy = occupy[unq_idx]
        else:
            pooled_occupy = None

        if self.fea_compre:
            processed_pooled_data = self.fea_compression(pooled_data)
            if occupy is not None:
                processed_pooled_data = processed_pooled_data * pooled_occupy
        else:
            processed_pooled_data = pooled_data

        return unq, processed_pooled_data, pooled_occupy