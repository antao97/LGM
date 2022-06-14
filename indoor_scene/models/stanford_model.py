''' Modified model for dynamics-aware attack on the Stanford dataset (S3DIS)

Dynamics-aware Adversarial Attack of 3D Sparse Convolution Network

Author: An Tao, Pengliang Ji
Email: ta19@mails.tsinghua.edu.cn, jpl1723@buaa.edu.cn
Date: 2022/1/13

Note:
    1: There exist some bugs in MinkowskiEngine v0.4.3, so we cannot directly multiply occupancy values with sparse convolution results. 
       We achieve this by constructing some functions to ensure the correctness. (See dynamics_aware_utils.py)
    2: Beause Equation (20) is applied in devoxelization, we achieve this by not multiplying occupancy value on the final layer output of the network.
       So we use the original self.final() forward function.  

'''

import torch
import torch.nn.functional as F
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiOps as me
from models.res16unet import Res16UNet34
from models.dynamics_aware_utils import conv_nosample, conv_downsample, conv_upample, fn_forward


def block_forward(block, x, occupy, real_num, stride=1):
    """
    Forward function for blocks in Res16UNet34 network
    """

    for i in range(len(block)):
        residual = x
        x = conv_nosample(block[i].conv1, x, occupy, real_num, stride=stride)
        x = fn_forward(block[i].norm1, x, occupy)
        x = block[i].relu(x)
        x = conv_nosample(block[i].conv2, x, occupy, real_num, stride=stride)
        x = fn_forward(block[i].norm2, x, occupy)
        if block[i].downsample is not None:
            residual = conv_nosample(block[i].downsample[0], residual, occupy, real_num, stride=stride)
            residual = fn_forward(block[i].downsample[1], residual, occupy)
        x._F += residual.F
        x = block[i].relu(x)
    return x


class NewRes16UNet34(Res16UNet34):
    """
    The modified Res16UNet34 network with dynamics-aware sparse convolutions
    """

    def forward(self, x, real_num, occupy):
        # out = self.conv0p1s1(x)
        out = conv_nosample(self.conv0p1s1, x, occupy, real_num)
        # out = self.bn0(out)
        out = fn_forward(self.bn0, out, occupy)
        out_p1 = self.relu(out)

        # out = self.conv1p1s2(out_p1)
        out, occupy_1, real_num_1 = conv_downsample(self.conv1p1s2, out_p1, occupy, real_num, out_stride=2)
        # out = self.bn1(out)
        out = fn_forward(self.bn1, out, occupy_1)
        out = self.relu(out)
        # out_b1p2 = self.block1(out)
        out_b1p2 = block_forward(self.block1, out, occupy_1, real_num_1, stride=2)

        # out = self.conv2p2s2(out_b1p2)
        out, occupy_2, real_num_2 = conv_downsample(self.conv2p2s2, out_b1p2, occupy_1, real_num_1, out_stride=4)
        # out = self.bn2(out)
        out = fn_forward(self.bn2, out, occupy_2)
        out = self.relu(out)
        # out_b2p4 = self.block2(out)
        out_b2p4 = block_forward(self.block2, out, occupy_2, real_num_2, stride=4)

        # out = self.conv3p4s2(out_b2p4)
        out, occupy_3, real_num_3 = conv_downsample(self.conv3p4s2, out_b2p4, occupy_2, real_num_2, out_stride=8)
        # out = self.bn3(out)
        out = fn_forward(self.bn3, out, occupy_3)
        out = self.relu(out)
        # out_b3p8 = self.block3(out)
        out_b3p8 = block_forward(self.block3, out, occupy_3, real_num_3, stride=8)

        # pixel_dist=16
        # out = self.conv4p8s2(out_b3p8)
        out, occupy_4, real_num_4 = conv_downsample(self.conv4p8s2, out_b3p8, occupy_3, real_num_3, out_stride=16)
        # out = self.bn4(out)
        out = fn_forward(self.bn4, out, occupy_4)
        out = self.relu(out)
        # out = self.block4(out)
        out = block_forward(self.block4, out, occupy_4, real_num_4, stride=16)

        # pixel_dist=8
        # out = self.convtr4p16s2(out)
        out = conv_upample(self.convtr4p16s2, out, occupy_4, real_num_4, out_b3p8.C, occupy_3, out_stride=8)
        # out = self.bntr4(out)
        out = fn_forward(self.bntr4, out, occupy_3)
        out = self.relu(out)

        # out = me.cat(out, out_b3p8)
        out._F = torch.cat([out.F, out_b3p8.F], dim=-1)
        # out = self.block5(out)
        out = block_forward(self.block5, out, occupy_3, real_num_3, stride=8)

        # pixel_dist=4
        # out = self.convtr5p8s2(out)
        out = conv_upample(self.convtr5p8s2, out, occupy_3, real_num_3, out_b2p4.C, occupy_2, out_stride=4)
        # out = self.bntr5(out)
        out = fn_forward(self.bntr5, out, occupy_2)
        out = self.relu(out)

        # out = me.cat(out, out_b2p4)
        out._F = torch.cat([out.F, out_b2p4.F], dim=-1)
        # out = self.block6(out)
        out = block_forward(self.block6, out, occupy_2, real_num_2, stride=4)

        # pixel_dist=2
        # out = self.convtr6p4s2(out)
        out = conv_upample(self.convtr6p4s2, out, occupy_2, real_num_2, out_b1p2.C, occupy_1, out_stride=2)
        # out = self.bntr6(out)
        out = fn_forward(self.bntr6, out, occupy_1)
        out = self.relu(out)

        # out = me.cat(out, out_b1p2)
        out._F = torch.cat([out.F, out_b1p2.F], dim=-1)
        # out = self.block7(out)
        out = block_forward(self.block7, out, occupy_1, real_num_1, stride=2)

        # pixel_dist=1
        # out = self.convtr7p2s2(out)
        out = conv_upample(self.convtr7p2s2, out, occupy_1, real_num_1, out_p1.C, occupy, out_stride=1)
        # out = self.bntr7(out)
        out = fn_forward(self.bntr7, out, occupy)
        out = self.relu(out)

        # out = me.cat(out, out_p1)
        out._F = torch.cat([out.F, out_p1.F], dim=-1)
        # out = self.block8(out)
        out = block_forward(self.block8, out, occupy, real_num)
        
        out = self.final(out)
        return out


class NewRes16UNet34_d2_1(Res16UNet34):
    """
    The modified Res16UNet34 network with dynamics-aware sparse convolutions for 2 GPU cards.

    Part 1
    """

    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        super(Res16UNet34, self).__init__(in_channels, out_channels, config, D)
        self.final = None

    def forward(self, x, real_num, occupy):
        # out = self.conv0p1s1(x)
        out = conv_nosample(self.conv0p1s1, x, occupy, real_num)
        # out = self.bn0(out)
        out = fn_forward(self.bn0, out, occupy)
        out_p1 = self.relu(out)

        # out = self.conv1p1s2(out_p1)
        out, occupy_1, real_num_1 = conv_downsample(self.conv1p1s2, out_p1, occupy, real_num, out_stride=2)
        # out = self.bn1(out)
        out = fn_forward(self.bn1, out, occupy_1)
        out = self.relu(out)
        # out_b1p2 = self.block1(out)
        out_b1p2 = block_forward(self.block1, out, occupy_1, real_num_1, stride=2)

        # out = self.conv2p2s2(out_b1p2)
        out, occupy_2, real_num_2 = conv_downsample(self.conv2p2s2, out_b1p2, occupy_1, real_num_1, out_stride=4)
        # out = self.bn2(out)
        out = fn_forward(self.bn2, out, occupy_2)
        out = self.relu(out)
        # out_b2p4 = self.block2(out)
        out_b2p4 = block_forward(self.block2, out, occupy_2, real_num_2, stride=4)

        # out = self.conv3p4s2(out_b2p4)
        out, occupy_3, real_num_3 = conv_downsample(self.conv3p4s2, out_b2p4, occupy_2, real_num_2, out_stride=8)
        # out = self.bn3(out)
        out = fn_forward(self.bn3, out, occupy_3)
        out = self.relu(out)
        # out_b3p8 = self.block3(out)
        out_b3p8 = block_forward(self.block3, out, occupy_3, real_num_3, stride=8)

        # pixel_dist=16
        # out = self.conv4p8s2(out_b3p8)
        out, occupy_4, real_num_4 = conv_downsample(self.conv4p8s2, out_b3p8, occupy_3, real_num_3, out_stride=16)
        # out = self.bn4(out)
        out = fn_forward(self.bn4, out, occupy_4)
        out = self.relu(out)
        # out = self.block4(out)
        out = block_forward(self.block4, out, occupy_4, real_num_4, stride=16)

        # pixel_dist=8
        # out = self.convtr4p16s2(out)
        out = conv_upample(self.convtr4p16s2, out, occupy_4, real_num_4, out_b3p8.C, occupy_3, out_stride=8)
        # out = self.bntr4(out)
        out = fn_forward(self.bntr4, out, occupy_3)
        out = self.relu(out)

        # out = me.cat(out, out_b3p8)
        out._F = torch.cat([out.F, out_b3p8.F], dim=-1)
        # out = self.block5(out)
        out = block_forward(self.block5, out, occupy_3, real_num_3, stride=8)

        # pixel_dist=4
        # out = self.convtr5p8s2(out)
        out = conv_upample(self.convtr5p8s2, out, occupy_3, real_num_3, out_b2p4.C, occupy_2, out_stride=4)
        # out = self.bntr5(out)
        out = fn_forward(self.bntr5, out, occupy_2)
        out = self.relu(out)

        # out = me.cat(out, out_b2p4)
        out._F = torch.cat([out.F, out_b2p4.F], dim=-1)
        # out = self.block6(out)
        out = block_forward(self.block6, out, occupy_2, real_num_2, stride=4)

        # pixel_dist=2
        # out = self.convtr6p4s2(out)
        out = conv_upample(self.convtr6p4s2, out, occupy_2, real_num_2, out_b1p2.C, occupy_1, out_stride=2)
        # out = self.bntr6(out)
        out = fn_forward(self.bntr6, out, occupy_1)
        out = self.relu(out)

        # out = me.cat(out, out_b1p2)
        out._F = torch.cat([out.F, out_b1p2.F], dim=-1)
        # out = self.block7(out)
        out = block_forward(self.block7, out, occupy_1, real_num_1, stride=2)

        # pixel_dist=1
        # out = self.convtr7p2s2(out)
        out = conv_upample(self.convtr7p2s2, out, occupy_1, real_num_1, out_p1.C, occupy, out_stride=1)
        # out = self.bntr7(out)
        out = fn_forward(self.bntr7, out, occupy)
        out = self.relu(out)

        # out = me.cat(out, out_p1)
        out._F = torch.cat([out.F, out_p1.F], dim=-1)

        residual = out
        # x = block[i].conv1(x)
        out = conv_nosample(self.block8[0].conv1, out, occupy, real_num)
        # x = block[i].norm1(x)
        out = fn_forward(self.block8[0].norm1, out, occupy)
        out = self.block8[0].relu(out)

        return [occupy, real_num, residual, out]


class NewRes16UNet34_d2_2(Res16UNet34):
    """
    The modified Res16UNet34 network with dynamics-aware sparse convolutions for 2 GPU cards.
    
    Part 2
    """

    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        super(Res16UNet34, self).__init__(in_channels, out_channels, config, D)
        self.conv0p1s1 = None
        self.bn0 = None
        self.conv1p1s2 = None
        self.bn1 = None
        self.block1 = None
        self.conv2p2s2 = None
        self.bn2 = None
        self.block2 = None
        self.conv3p4s2 = None
        self.bn3 = None
        self.block3 = None
        self.conv4p8s2 = None
        self.bn4 = None
        self.block4 = None
        self.convtr4p16s2 = None
        self.bntr4 = None
        self.block5 = None
        self.convtr5p8s2 = None
        self.bntr5 = None
        self.block6 = None
        self.convtr6p4s2 = None
        self.bntr6 = None
        self.block7 = None
        self.convtr7p2s2 = None
        self.bntr7 = None

    def forward(self, interm):
        [occupy, real_num, residual, out] = interm
        occupy, residual, out = \
            occupy.to('cuda'), \
            ME.SparseTensor(residual.F, residual.C, tensor_stride=residual.tensor_stride).to('cuda'), \
            ME.SparseTensor(out.F, out.C, tensor_stride=out.tensor_stride).to('cuda')

        # x = block[i].conv2(x)
        out = conv_nosample(self.block8[0].conv2, out, occupy, real_num)
        # x = block[i].norm2(x)
        out = fn_forward(self.block8[0].norm2, out, occupy)
        if self.block8[0].downsample is not None:
            # residual = block[i].downsample[0](residual)
            residual = conv_nosample(self.block8[0].downsample[0], residual, occupy, real_num)
            # residual = block[i].downsample[1](residual)
            residual = fn_forward(self.block8[0].downsample[1], residual, occupy)
        out._F = out.F + residual.F
        out = self.block8[0].relu(out)
        
        # out = self.block8(out)
        out = block_forward([self.block8[1]], out, occupy, real_num)
        
        out = self.final(out)
        return out


class NewRes16UNet34_d4_1(Res16UNet34):
    """
    The modified Res16UNet34 network with dynamics-aware sparse convolutions for 4 GPU cards.
    
    Part 1
    """

    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        super(Res16UNet34, self).__init__(in_channels, out_channels, config, D)
        self.convtr7p2s2 = None
        self.bntr7 = None
        self.block8 = None
        self.final = None

    def forward(self, x, real_num, occupy):
        # out = self.conv0p1s1(x)
        out = conv_nosample(self.conv0p1s1, x, occupy, real_num)
        # out = self.bn0(out)
        out = fn_forward(self.bn0, out, occupy)
        out_p1 = self.relu(out)

        # out = self.conv1p1s2(out_p1)
        out, occupy_1, real_num_1 = conv_downsample(self.conv1p1s2, out_p1, occupy, real_num, out_stride=2)
        # out = self.bn1(out)
        out = fn_forward(self.bn1, out, occupy_1)
        out = self.relu(out)
        # out_b1p2 = self.block1(out)
        out_b1p2 = block_forward(self.block1, out, occupy_1, real_num_1, stride=2)

        # out = self.conv2p2s2(out_b1p2)
        out, occupy_2, real_num_2 = conv_downsample(self.conv2p2s2, out_b1p2, occupy_1, real_num_1, out_stride=4)
        # out = self.bn2(out)
        out = fn_forward(self.bn2, out, occupy_2)
        out = self.relu(out)
        # out_b2p4 = self.block2(out)
        out_b2p4 = block_forward(self.block2, out, occupy_2, real_num_2, stride=4)

        # out = self.conv3p4s2(out_b2p4)
        out, occupy_3, real_num_3 = conv_downsample(self.conv3p4s2, out_b2p4, occupy_2, real_num_2, out_stride=8)
        # out = self.bn3(out)
        out = fn_forward(self.bn3, out, occupy_3)
        out = self.relu(out)
        # out_b3p8 = self.block3(out)
        out_b3p8 = block_forward(self.block3, out, occupy_3, real_num_3, stride=8)

        # pixel_dist=16
        # out = self.conv4p8s2(out_b3p8)
        out, occupy_4, real_num_4 = conv_downsample(self.conv4p8s2, out_b3p8, occupy_3, real_num_3, out_stride=16)
        # out = self.bn4(out)
        out = fn_forward(self.bn4, out, occupy_4)
        out = self.relu(out)
        # out = self.block4(out)
        out = block_forward(self.block4, out, occupy_4, real_num_4, stride=16)

        # pixel_dist=8
        # out = self.convtr4p16s2(out)
        out = conv_upample(self.convtr4p16s2, out, occupy_4, real_num_4, out_b3p8.C, occupy_3, out_stride=8)
        # out = self.bntr4(out)
        out = fn_forward(self.bntr4, out, occupy_3)
        out = self.relu(out)

        # out = me.cat(out, out_b3p8)
        out._F = torch.cat([out.F, out_b3p8.F], dim=-1)
        # out = self.block5(out)
        out = block_forward(self.block5, out, occupy_3, real_num_3, stride=8)

        # pixel_dist=4
        # out = self.convtr5p8s2(out)
        out = conv_upample(self.convtr5p8s2, out, occupy_3, real_num_3, out_b2p4.C, occupy_2, out_stride=4)
        # out = self.bntr5(out)
        out = fn_forward(self.bntr5, out, occupy_2)
        out = self.relu(out)

        # out = me.cat(out, out_b2p4)
        out._F = torch.cat([out.F, out_b2p4.F], dim=-1)
        # out = self.block6(out)
        out = block_forward(self.block6, out, occupy_2, real_num_2, stride=4)

        # pixel_dist=2
        # out = self.convtr6p4s2(out)
        out = conv_upample(self.convtr6p4s2, out, occupy_2, real_num_2, out_b1p2.C, occupy_1, out_stride=2)
        # out = self.bntr6(out)
        out = fn_forward(self.bntr6, out, occupy_1)
        out = self.relu(out)

        # out = me.cat(out, out_b1p2)
        out._F = torch.cat([out.F, out_b1p2.F], dim=-1)
        # out = self.block7(out)
        out = block_forward([self.block7[0]], out, occupy_1, real_num_1, stride=2)

        return [[occupy, real_num, out_p1], [occupy_1, real_num_1, out]]


class NewRes16UNet34_d4_2(Res16UNet34):
    """
    The modified Res16UNet34 network with dynamics-aware sparse convolutions for 4 GPU cards.
    
    Part 2
    """

    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        super(Res16UNet34, self).__init__(in_channels, out_channels, config, D)
        self.conv0p1s1 = None
        self.bn0 = None
        self.conv1p1s2 = None
        self.bn1 = None
        self.block1 = None
        self.conv2p2s2 = None
        self.bn2 = None
        self.block2 = None
        self.conv3p4s2 = None
        self.bn3 = None
        self.block3 = None
        self.conv4p8s2 = None
        self.bn4 = None
        self.block4 = None
        self.convtr4p16s2 = None
        self.bntr4 = None
        self.block5 = None
        self.convtr5p8s2 = None
        self.bntr5 = None
        self.block6 = None
        self.convtr6p4s2 = None
        self.bntr6 = None
        self.final = None

    def forward(self, interm):
        [[occupy, real_num, out_p1], [occupy_1, real_num_1, out]] = interm
        occupy, out_p1, occupy_1, out = \
            occupy.to('cuda'), ME.SparseTensor(out_p1.F, out_p1.C, tensor_stride=out_p1.tensor_stride).to('cuda'), \
            occupy_1.to('cuda'), ME.SparseTensor(out.F, out.C, tensor_stride=out.tensor_stride).to('cuda')

        out = block_forward([self.block7[1]], out, occupy_1, real_num_1, stride=2)

        # pixel_dist=1
        # out = self.convtr7p2s2(out)
        out = conv_upample(self.convtr7p2s2, out, occupy_1, real_num_1, out_p1.C, occupy, out_stride=1)
        # out = self.bntr7(out)
        out = fn_forward(self.bntr7, out, occupy)
        out = self.relu(out)

        # out = me.cat(out, out_p1)
        out._F = torch.cat([out.F, out_p1.F], dim=-1)

        residual = out
        # x = block[i].conv1(x)
        out = conv_nosample(self.block8[0].conv1, out, occupy, real_num)
        # x = block[i].norm1(x)
        out = fn_forward(self.block8[0].norm1, out, occupy)
        out = self.block8[0].relu(out)

        return [occupy, real_num, residual, out]


class NewRes16UNet34_d4_3(Res16UNet34):
    """
    The modified Res16UNet34 network with dynamics-aware sparse convolutions for 4 GPU cards.
    
    Part 3
    """

    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        super(Res16UNet34, self).__init__(in_channels, out_channels, config, D)
        self.conv0p1s1 = None
        self.bn0 = None
        self.conv1p1s2 = None
        self.bn1 = None
        self.block1 = None
        self.conv2p2s2 = None
        self.bn2 = None
        self.block2 = None
        self.conv3p4s2 = None
        self.bn3 = None
        self.block3 = None
        self.conv4p8s2 = None
        self.bn4 = None
        self.block4 = None
        self.convtr4p16s2 = None
        self.bntr4 = None
        self.block5 = None
        self.convtr5p8s2 = None
        self.bntr5 = None
        self.block6 = None
        self.convtr6p4s2 = None
        self.bntr6 = None
        self.block7 = None
        self.convtr7p2s2 = None
        self.bntr7 = None
        self.final = None

    def forward(self, interm):
        [occupy, real_num, residual, out] = interm
        occupy, residual, out = \
            occupy.to('cuda'), ME.SparseTensor(residual.F, residual.C, tensor_stride=residual.tensor_stride).to('cuda'), \
            ME.SparseTensor(out.F, out.C, tensor_stride=out.tensor_stride).to('cuda')
        
        # out = self.block8(out)
        # x = block[i].conv2(x)
        out = conv_nosample(self.block8[0].conv2, out, occupy, real_num)
        # x = block[i].norm2(x)
        out = fn_forward(self.block8[0].norm2, out, occupy)
        if self.block8[0].downsample is not None:
            # residual = block[i].downsample[0](residual)
            residual = conv_nosample(self.block8[0].downsample[0], residual, occupy, real_num)
            # residual = block[i].downsample[1](residual)
            residual = fn_forward(self.block8[0].downsample[1], residual, occupy)
        out._F = out.F + residual.F
        out = self.block8[0].relu(out)

        return [occupy, real_num, out]


class NewRes16UNet34_d4_4(Res16UNet34):
    """
    The modified Res16UNet34 network with dynamics-aware sparse convolutions for 4 GPU cards.
    
    Part 4
    """

    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        super(Res16UNet34, self).__init__(in_channels, out_channels, config, D)
        self.conv0p1s1 = None
        self.bn0 = None
        self.conv1p1s2 = None
        self.bn1 = None
        self.block1 = None
        self.conv2p2s2 = None
        self.bn2 = None
        self.block2 = None
        self.conv3p4s2 = None
        self.bn3 = None
        self.block3 = None
        self.conv4p8s2 = None
        self.bn4 = None
        self.block4 = None
        self.convtr4p16s2 = None
        self.bntr4 = None
        self.block5 = None
        self.convtr5p8s2 = None
        self.bntr5 = None
        self.block6 = None
        self.convtr6p4s2 = None
        self.bntr6 = None
        self.block7 = None
        self.convtr7p2s2 = None
        self.bntr7 = None

    def forward(self, interm):
        [occupy, real_num, out] = interm
        occupy, out = \
            occupy.to('cuda'), ME.SparseTensor(out.F, out.C).to('cuda')

        out = block_forward([self.block8[1]], out, occupy, real_num)
        
        out = self.final(out)
        return out


class NewRes16UNet34_d8_1(Res16UNet34):
    """
    The modified Res16UNet34 network with dynamics-aware sparse convolutions for 8 GPU cards.
    
    Part 1
    """

    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        super(Res16UNet34, self).__init__(in_channels, out_channels, config, D)
        self.convtr4p16s2 = None
        self.bntr4 = None
        self.block5 = None
        self.convtr5p8s2 = None
        self.bntr5 = None
        self.block6 = None
        self.convtr6p4s2 = None
        self.bntr6 = None
        self.block7 = None
        self.convtr7p2s2 = None
        self.bntr7 = None
        self.block8 = None
        self.final = None

    def forward(self, x, real_num, occupy):
        # out = self.conv0p1s1(x)
        out = conv_nosample(self.conv0p1s1, x, occupy, real_num)
        # out = self.bn0(out)
        out = fn_forward(self.bn0, out, occupy)
        out_p1 = self.relu(out)

        # out = self.conv1p1s2(out_p1)
        out, occupy_1, real_num_1 = conv_downsample(self.conv1p1s2, out_p1, occupy, real_num, out_stride=2)
        # out = self.bn1(out)
        out = fn_forward(self.bn1, out, occupy_1)
        out = self.relu(out)
        # out_b1p2 = self.block1(out)
        out_b1p2 = block_forward(self.block1, out, occupy_1, real_num_1, stride=2)

        # out = self.conv2p2s2(out_b1p2)
        out, occupy_2, real_num_2 = conv_downsample(self.conv2p2s2, out_b1p2, occupy_1, real_num_1, out_stride=4)
        # out = self.bn2(out)
        out = fn_forward(self.bn2, out, occupy_2)
        out = self.relu(out)
        # out_b2p4 = self.block2(out)
        out_b2p4 = block_forward(self.block2, out, occupy_2, real_num_2, stride=4)

        # out = self.conv3p4s2(out_b2p4)
        out, occupy_3, real_num_3 = conv_downsample(self.conv3p4s2, out_b2p4, occupy_2, real_num_2, out_stride=8)
        # out = self.bn3(out)
        out = fn_forward(self.bn3, out, occupy_3)
        out = self.relu(out)
        # out_b3p8 = self.block3(out)
        out_b3p8 = block_forward(self.block3, out, occupy_3, real_num_3, stride=8)

        # pixel_dist=16
        # out = self.conv4p8s2(out_b3p8)
        out, occupy_4, real_num_4 = conv_downsample(self.conv4p8s2, out_b3p8, occupy_3, real_num_3, out_stride=16)
        # out = self.bn4(out)
        out = fn_forward(self.bn4, out, occupy_4)
        out = self.relu(out)
        # out = self.block4(out)
        out = block_forward(self.block4, out, occupy_4, real_num_4, stride=16)

        return [[occupy, real_num, out_p1], [occupy_1, real_num_1, out_b1p2], [occupy_2, real_num_2, out_b2p4], \
            [occupy_3, real_num_3, out_b3p8], [occupy_4, real_num_4, out]]


class NewRes16UNet34_d8_2(Res16UNet34):
    """
    The modified Res16UNet34 network with dynamics-aware sparse convolutions for 8 GPU cards.
    
    Part 2
    """

    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        super(Res16UNet34, self).__init__(in_channels, out_channels, config, D)
        self.conv0p1s1 = None
        self.bn0 = None
        self.conv1p1s2 = None
        self.bn1 = None
        self.block1 = None
        self.conv2p2s2 = None
        self.bn2 = None
        self.block2 = None
        self.conv3p4s2 = None
        self.bn3 = None
        self.block3 = None
        self.conv4p8s2 = None
        self.bn4 = None
        self.block4 = None
        self.convtr7p2s2 = None
        self.bntr7 = None
        self.block8 = None
        self.final = None

    def forward(self, interm):
        [[occupy, real_num, out_p1], [occupy_1, real_num_1, out_b1p2], [occupy_2, real_num_2, out_b2p4], \
            [occupy_3, real_num_3, out_b3p8], [occupy_4, real_num_4, out]] = interm
        occupy, out_p1, occupy_1, out_b1p2, occupy_2, out_b2p4, occupy_3, out_b3p8, occupy_4, out = \
            occupy.to('cuda'), ME.SparseTensor(out_p1.F, out_p1.C, tensor_stride=out_p1.tensor_stride).to('cuda'), \
            occupy_1.to('cuda'), ME.SparseTensor(out_b1p2.F, out_b1p2.C, tensor_stride=out_b1p2.tensor_stride).to('cuda'), \
            occupy_2.to('cuda'), ME.SparseTensor(out_b2p4.F, out_b2p4.C, tensor_stride=out_b2p4.tensor_stride).to('cuda'), \
            occupy_3.to('cuda'), ME.SparseTensor(out_b3p8.F, out_b3p8.C, tensor_stride=out_b3p8.tensor_stride).to('cuda'), \
            occupy_4.to('cuda'), ME.SparseTensor(out.F, out.C, tensor_stride=out.tensor_stride).to('cuda')

        # pixel_dist=8
        # out = self.convtr4p16s2(out)
        out = conv_upample(self.convtr4p16s2, out, occupy_4, real_num_4, out_b3p8.C, occupy_3, out_stride=8)
        # out = self.bntr4(out)
        out = fn_forward(self.bntr4, out, occupy_3)
        out = self.relu(out)

        # out = me.cat(out, out_b3p8)
        out._F = torch.cat([out.F, out_b3p8.F], dim=-1)
        # out = self.block5(out)
        out = block_forward(self.block5, out, occupy_3, real_num_3, stride=8)

        # pixel_dist=4
        # out = self.convtr5p8s2(out)
        out = conv_upample(self.convtr5p8s2, out, occupy_3, real_num_3, out_b2p4.C, occupy_2, out_stride=4)
        # out = self.bntr5(out)
        out = fn_forward(self.bntr5, out, occupy_2)
        out = self.relu(out)

        # out = me.cat(out, out_b2p4)
        out._F = torch.cat([out.F, out_b2p4.F], dim=-1)
        # out = self.block6(out)
        out = block_forward(self.block6, out, occupy_2, real_num_2, stride=4)

        # pixel_dist=2
        # out = self.convtr6p4s2(out)
        out = conv_upample(self.convtr6p4s2, out, occupy_2, real_num_2, out_b1p2.C, occupy_1, out_stride=2)
        # out = self.bntr6(out)
        out = fn_forward(self.bntr6, out, occupy_1)
        out = self.relu(out)

        # out = me.cat(out, out_b1p2)
        out._F = torch.cat([out.F, out_b1p2.F], dim=-1)
        # out = self.block7(out)
        out = block_forward([self.block7[0]], out, occupy_1, real_num_1, stride=2)

        return [[occupy, real_num, out_p1], [occupy_1, real_num_1, out]]


class NewRes16UNet34_d8_3(Res16UNet34):
    """
    The modified Res16UNet34 network with dynamics-aware sparse convolutions for 8 GPU cards.
    
    Part 3
    """

    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        super(Res16UNet34, self).__init__(in_channels, out_channels, config, D)
        self.conv0p1s1 = None
        self.bn0 = None
        self.conv1p1s2 = None
        self.bn1 = None
        self.block1 = None
        self.conv2p2s2 = None
        self.bn2 = None
        self.block2 = None
        self.conv3p4s2 = None
        self.bn3 = None
        self.block3 = None
        self.conv4p8s2 = None
        self.bn4 = None
        self.block4 = None
        self.convtr4p16s2 = None
        self.bntr4 = None
        self.block5 = None
        self.convtr5p8s2 = None
        self.bntr5 = None
        self.block6 = None
        self.convtr6p4s2 = None
        self.bntr6 = None
        self.block8 = None
        self.final = None

    def forward(self, interm):
        [[occupy, real_num, out_p1], [occupy_1, real_num_1, out]] = interm
        occupy, out_p1, occupy_1, out = \
            occupy.to('cuda'), ME.SparseTensor(out_p1.F, out_p1.C, tensor_stride=out_p1.tensor_stride).to('cuda'), \
            occupy_1.to('cuda'), ME.SparseTensor(out.F, out.C, tensor_stride=out.tensor_stride).to('cuda')

        out = block_forward([self.block7[1]], out, occupy_1, real_num_1, stride=2)

        # pixel_dist=1
        # out = self.convtr7p2s2(out)
        out = conv_upample(self.convtr7p2s2, out, occupy_1, real_num_1, out_p1.C, occupy, out_stride=1)
        # out = self.bntr7(out)
        out = fn_forward(self.bntr7, out, occupy)
        out = self.relu(out)

        # out = me.cat(out, out_p1)
        out._F = torch.cat([out.F, out_p1.F], dim=-1)

        return [occupy, real_num, out]


class NewRes16UNet34_d8_4(Res16UNet34):
    """
    The modified Res16UNet34 network with dynamics-aware sparse convolutions for 8 GPU cards.
    
    Part 4
    """

    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        super(Res16UNet34, self).__init__(in_channels, out_channels, config, D)
        self.conv0p1s1 = None
        self.bn0 = None
        self.conv1p1s2 = None
        self.bn1 = None
        self.block1 = None
        self.conv2p2s2 = None
        self.bn2 = None
        self.block2 = None
        self.conv3p4s2 = None
        self.bn3 = None
        self.block3 = None
        self.conv4p8s2 = None
        self.bn4 = None
        self.block4 = None
        self.convtr4p16s2 = None
        self.bntr4 = None
        self.block5 = None
        self.convtr5p8s2 = None
        self.bntr5 = None
        self.block6 = None
        self.convtr6p4s2 = None
        self.bntr6 = None
        self.block7 = None
        self.convtr7p2s2 = None
        self.bntr7 = None
        self.final = None

    def forward(self, interm):
        [occupy, real_num, out] = interm
        occupy, out = \
            occupy.to('cuda'), ME.SparseTensor(out.F, out.C, tensor_stride=out.tensor_stride).to('cuda')

        residual = out
        # x = block[i].conv1(x)
        out = conv_nosample(self.block8[0].conv1, out, occupy, real_num)
        # x = block[i].norm1(x)
        out = fn_forward(self.block8[0].norm1, out, occupy)
        out = self.block8[0].relu(out)

        return [occupy, real_num, residual, out]


class NewRes16UNet34_d8_5(Res16UNet34):
    """
    The modified Res16UNet34 network with dynamics-aware sparse convolutions for 8 GPU cards.
    
    Part 5
    """

    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        super(Res16UNet34, self).__init__(in_channels, out_channels, config, D)
        self.conv0p1s1 = None
        self.bn0 = None
        self.conv1p1s2 = None
        self.bn1 = None
        self.block1 = None
        self.conv2p2s2 = None
        self.bn2 = None
        self.block2 = None
        self.conv3p4s2 = None
        self.bn3 = None
        self.block3 = None
        self.conv4p8s2 = None
        self.bn4 = None
        self.block4 = None
        self.convtr4p16s2 = None
        self.bntr4 = None
        self.block5 = None
        self.convtr5p8s2 = None
        self.bntr5 = None
        self.block6 = None
        self.convtr6p4s2 = None
        self.bntr6 = None
        self.block7 = None
        self.convtr7p2s2 = None
        self.bntr7 = None
        self.final = None

    def forward(self, interm):
        [occupy, real_num, residual, out] = interm
        occupy, residual, out = \
            occupy.to('cuda'), ME.SparseTensor(residual.F, residual.C, tensor_stride=residual.tensor_stride).to('cuda'), \
            ME.SparseTensor(out.F, out.C, tensor_stride=out.tensor_stride).to('cuda')
        
        # out = self.block8(out)
        # x = block[i].conv2(x)
        out = conv_nosample(self.block8[0].conv2, out, occupy, real_num)
        # x = block[i].norm2(x)
        out = fn_forward(self.block8[0].norm2, out, occupy)

        return [occupy, real_num, residual, out]


class NewRes16UNet34_d8_6(Res16UNet34):
    """
    The modified Res16UNet34 network with dynamics-aware sparse convolutions for 8 GPU cards.
    
    Part 6
    """

    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        super(Res16UNet34, self).__init__(in_channels, out_channels, config, D)
        self.conv0p1s1 = None
        self.bn0 = None
        self.conv1p1s2 = None
        self.bn1 = None
        self.block1 = None
        self.conv2p2s2 = None
        self.bn2 = None
        self.block2 = None
        self.conv3p4s2 = None
        self.bn3 = None
        self.block3 = None
        self.conv4p8s2 = None
        self.bn4 = None
        self.block4 = None
        self.convtr4p16s2 = None
        self.bntr4 = None
        self.block5 = None
        self.convtr5p8s2 = None
        self.bntr5 = None
        self.block6 = None
        self.convtr6p4s2 = None
        self.bntr6 = None
        self.block7 = None
        self.convtr7p2s2 = None
        self.bntr7 = None
        self.final = None

    def forward(self, interm):
        [occupy, real_num, residual, out] = interm
        occupy, residual, out = \
            occupy.to('cuda'), ME.SparseTensor(residual.F, residual.C, tensor_stride=residual.tensor_stride).to('cuda'), \
            ME.SparseTensor(out.F, out.C, tensor_stride=out.tensor_stride).to('cuda')
        
        if self.block8[0].downsample is not None:
            # residual = block[i].downsample[0](residual)
            residual = conv_nosample(self.block8[0].downsample[0], residual, occupy, real_num)
            # residual = block[i].downsample[1](residual)
            residual = fn_forward(self.block8[0].downsample[1], residual, occupy)
        out._F = out.F + residual.F
        out = self.block8[0].relu(out)

        return [occupy, real_num, out]


class NewRes16UNet34_d8_7(Res16UNet34):
    """
    The modified Res16UNet34 network with dynamics-aware sparse convolutions for 8 GPU cards.
    
    Part 7
    """

    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        super(Res16UNet34, self).__init__(in_channels, out_channels, config, D)
        self.conv0p1s1 = None
        self.bn0 = None
        self.conv1p1s2 = None
        self.bn1 = None
        self.block1 = None
        self.conv2p2s2 = None
        self.bn2 = None
        self.block2 = None
        self.conv3p4s2 = None
        self.bn3 = None
        self.block3 = None
        self.conv4p8s2 = None
        self.bn4 = None
        self.block4 = None
        self.convtr4p16s2 = None
        self.bntr4 = None
        self.block5 = None
        self.convtr5p8s2 = None
        self.bntr5 = None
        self.block6 = None
        self.convtr6p4s2 = None
        self.bntr6 = None
        self.block7 = None
        self.convtr7p2s2 = None
        self.bntr7 = None
        self.final = None

    def forward(self, interm):
        [occupy, real_num, out] = interm
        occupy, out = \
            occupy.to('cuda'), ME.SparseTensor(out.F, out.C).to('cuda')

        out = block_forward([self.block8[1]], out, occupy, real_num)
        
        return [occupy, real_num, out]


class NewRes16UNet34_d8_8(Res16UNet34):
    """
    The modified Res16UNet34 network with dynamics-aware sparse convolutions for 8 GPU cards.
    
    Part 8
    """

    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        super(Res16UNet34, self).__init__(in_channels, out_channels, config, D)
        self.conv0p1s1 = None
        self.bn0 = None
        self.conv1p1s2 = None
        self.bn1 = None
        self.block1 = None
        self.conv2p2s2 = None
        self.bn2 = None
        self.block2 = None
        self.conv3p4s2 = None
        self.bn3 = None
        self.block3 = None
        self.conv4p8s2 = None
        self.bn4 = None
        self.block4 = None
        self.convtr4p16s2 = None
        self.bntr4 = None
        self.block5 = None
        self.convtr5p8s2 = None
        self.bntr5 = None
        self.block6 = None
        self.convtr6p4s2 = None
        self.bntr6 = None
        self.block7 = None
        self.convtr7p2s2 = None
        self.bntr7 = None
        self.block8 = None

    def forward(self, interm):
        [occupy, real_num, out] = interm
        occupy, out = \
            occupy.to('cuda'), ME.SparseTensor(out.F, out.C).to('cuda')
        
        out = self.final(out)
        return out

