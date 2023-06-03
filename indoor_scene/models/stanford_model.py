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
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiOps as me
from models.res16unet import Res16UNet34
from models.dynamics_aware_utils import conv_nosample, conv_downsample, conv_upample, fn_forward, conv_flops


# Note:
#   1: There exist some bugs in MinkowskiEngine v0.4.3, so we cannot directly multiply occupancy values with sparse convolution results. 
#      We achieve this by constructing some functions to ensure the correctness.
#   2: Beause Equation (20) is applied in devoxelization, we achieve this by not multiplying occupancy value on the final layer output of the network.
#      So we use the original self.final() forward function.  


def block_forward(block, x, occupy, real_num, stride=1, flops=0, mul_occupy_after=True):
    """
    Forward function for blocks in Res16UNet34 network
    """

    for i in range(len(block)):
        residual = x
        x, flops = conv_nosample(block[i].conv1, x, occupy, real_num, stride=stride, flops=flops, mul_occupy_after=mul_occupy_after)
        x, flops = fn_forward(block[i].norm1, x, occupy, flops=flops, mul_occupy_after=mul_occupy_after)
        x = block[i].relu(x)
        x, flops = conv_nosample(block[i].conv2, x, occupy, real_num, stride=stride, flops=flops, mul_occupy_after=mul_occupy_after)
        x, flops = fn_forward(block[i].norm2, x, occupy, flops=flops, mul_occupy_after=mul_occupy_after)
        if block[i].downsample is not None:
            residual, flops = conv_nosample(block[i].downsample[0], residual, occupy, real_num, stride=stride, flops=flops, mul_occupy_after=mul_occupy_after)
            residual, flops = fn_forward(block[i].downsample[1], residual, occupy, flops=flops, mul_occupy_after=mul_occupy_after)
        x._F += residual.F
        x = block[i].relu(x)
    return x, flops



class NewRes16UNet34(Res16UNet34):
    """
    The modified Res16UNet34 network with dynamics-aware sparse convolutions
    """
    def __init__(self, in_channels, out_channels, config, D=3, dynamics_aware=True, **kwargs):
        super(Res16UNet34, self).__init__(in_channels, out_channels, config, D)
        self.dynamics_aware = dynamics_aware

    def forward(self, x, real_num, occupy=None):
        # out = self.conv0p1s1(x)
        out, flops = conv_nosample(self.conv0p1s1, x, occupy, real_num, mul_occupy_after=self.dynamics_aware)
        # out = self.bn0(out)
        out, flops = fn_forward(self.bn0, out, occupy, flops=flops, mul_occupy_after=self.dynamics_aware)
        out_p1 = self.relu(out)

        # out = self.conv1p1s2(out_p1)
        out, occupy_1, real_num_1, flops = conv_downsample(self.conv1p1s2, out_p1, occupy, real_num, out_stride=2, flops=flops, mul_occupy_after=self.dynamics_aware)
        # out = self.bn1(out)
        out, flops = fn_forward(self.bn1, out, occupy_1, flops=flops, mul_occupy_after=self.dynamics_aware)
        out = self.relu(out)
        # out_b1p2 = self.block1(out)
        out_b1p2, flops = block_forward(self.block1, out, occupy_1, real_num_1, stride=2, flops=flops, mul_occupy_after=self.dynamics_aware)

        # out = self.conv2p2s2(out_b1p2)
        out, occupy_2, real_num_2, flops = conv_downsample(self.conv2p2s2, out_b1p2, occupy_1, real_num_1, out_stride=4, flops=flops, mul_occupy_after=self.dynamics_aware)
        # out = self.bn2(out)
        out, flops = fn_forward(self.bn2, out, occupy_2, flops=flops, mul_occupy_after=self.dynamics_aware)
        out = self.relu(out)
        # out_b2p4 = self.block2(out)
        out_b2p4, flops = block_forward(self.block2, out, occupy_2, real_num_2, stride=4, flops=flops, mul_occupy_after=self.dynamics_aware)

        # out = self.conv3p4s2(out_b2p4)
        out, occupy_3, real_num_3, flops = conv_downsample(self.conv3p4s2, out_b2p4, occupy_2, real_num_2, out_stride=8, flops=flops, mul_occupy_after=self.dynamics_aware)
        # out = self.bn3(out)
        out, flops = fn_forward(self.bn3, out, occupy_3, flops=flops, mul_occupy_after=self.dynamics_aware)
        out = self.relu(out)
        # out_b3p8 = self.block3(out)
        out_b3p8, flops = block_forward(self.block3, out, occupy_3, real_num_3, stride=8, flops=flops, mul_occupy_after=self.dynamics_aware)

        # pixel_dist=16
        # out = self.conv4p8s2(out_b3p8)
        out, occupy_4, real_num_4, flops = conv_downsample(self.conv4p8s2, out_b3p8, occupy_3, real_num_3, out_stride=16, flops=flops, mul_occupy_after=self.dynamics_aware)
        # out = self.bn4(out)
        out, flops = fn_forward(self.bn4, out, occupy_4, flops=flops, mul_occupy_after=self.dynamics_aware)
        out = self.relu(out)
        # out = self.block4(out)
        out, flops = block_forward(self.block4, out, occupy_4, real_num_4, stride=16, flops=flops, mul_occupy_after=self.dynamics_aware)

        # pixel_dist=8
        # out = self.convtr4p16s2(out)
        out, flops = conv_upample(self.convtr4p16s2, out, occupy_4, real_num_4, out_b3p8.C, occupy_3, out_stride=8, flops=flops, mul_occupy_after=self.dynamics_aware)
        # out = self.bntr4(out)
        out, flops = fn_forward(self.bntr4, out, occupy_3, flops=flops, mul_occupy_after=self.dynamics_aware)
        out = self.relu(out)

        # out = me.cat(out, out_b3p8)
        out._F = torch.cat([out.F, out_b3p8.F], dim=-1)
        # out = self.block5(out)
        out, flops = block_forward(self.block5, out, occupy_3, real_num_3, stride=8, flops=flops, mul_occupy_after=self.dynamics_aware)

        # pixel_dist=4
        # out = self.convtr5p8s2(out)
        out, flops = conv_upample(self.convtr5p8s2, out, occupy_3, real_num_3, out_b2p4.C, occupy_2, out_stride=4, flops=flops, mul_occupy_after=self.dynamics_aware)
        # out = self.bntr5(out)
        out, flops = fn_forward(self.bntr5, out, occupy_2, flops=flops, mul_occupy_after=self.dynamics_aware)
        out = self.relu(out)

        # out = me.cat(out, out_b2p4)
        out._F = torch.cat([out.F, out_b2p4.F], dim=-1)
        # out = self.block6(out)
        out, flops = block_forward(self.block6, out, occupy_2, real_num_2, stride=4, flops=flops, mul_occupy_after=self.dynamics_aware)

        # pixel_dist=2
        # out = self.convtr6p4s2(out)
        out, flops = conv_upample(self.convtr6p4s2, out, occupy_2, real_num_2, out_b1p2.C, occupy_1, out_stride=2, flops=flops, mul_occupy_after=self.dynamics_aware)
        # out = self.bntr6(out)
        out, flops = fn_forward(self.bntr6, out, occupy_1, flops=flops, mul_occupy_after=self.dynamics_aware)
        out = self.relu(out)

        # out = me.cat(out, out_b1p2)
        out._F = torch.cat([out.F, out_b1p2.F], dim=-1)
        # out = self.block7(out)
        out, flops = block_forward(self.block7, out, occupy_1, real_num_1, stride=2, flops=flops, mul_occupy_after=self.dynamics_aware)

        # pixel_dist=1
        # out = self.convtr7p2s2(out)
        out, flops = conv_upample(self.convtr7p2s2, out, occupy_1, real_num_1, out_p1.C, occupy, out_stride=1, flops=flops, mul_occupy_after=self.dynamics_aware)
        # out = self.bntr7(out)
        out, flops = fn_forward(self.bntr7, out, occupy, flops=flops, mul_occupy_after=self.dynamics_aware)
        out = self.relu(out)

        # out = me.cat(out, out_p1)
        out._F = torch.cat([out.F, out_p1.F], dim=-1)
        # out = self.block8(out)
        out, flops = block_forward(self.block8, out, occupy, real_num, flops=flops, mul_occupy_after=self.dynamics_aware)
        
        # out = self.final(out)
        out, flops = conv_nosample(self.final, out, occupy, real_num, flops=flops, mul_occupy_after=False)
        return out, flops


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
        out, flops = conv_nosample(self.conv0p1s1, x, occupy, real_num)
        # out = self.bn0(out)
        out, flops = fn_forward(self.bn0, out, occupy, flops=flops)
        out_p1 = self.relu(out)

        # out = self.conv1p1s2(out_p1)
        out, occupy_1, real_num_1, flops = conv_downsample(self.conv1p1s2, out_p1, occupy, real_num, out_stride=2, flops=flops)
        # out = self.bn1(out)
        out, flops = fn_forward(self.bn1, out, occupy_1, flops=flops)
        out = self.relu(out)
        # out_b1p2 = self.block1(out)
        out_b1p2, flops = block_forward(self.block1, out, occupy_1, real_num_1, stride=2, flops=flops)

        # out = self.conv2p2s2(out_b1p2)
        out, occupy_2, real_num_2, flops = conv_downsample(self.conv2p2s2, out_b1p2, occupy_1, real_num_1, out_stride=4, flops=flops)
        # out = self.bn2(out)
        out, flops = fn_forward(self.bn2, out, occupy_2, flops=flops)
        out = self.relu(out)
        # out_b2p4 = self.block2(out)
        out_b2p4, flops = block_forward(self.block2, out, occupy_2, real_num_2, stride=4, flops=flops)

        # out = self.conv3p4s2(out_b2p4)
        out, occupy_3, real_num_3, flops = conv_downsample(self.conv3p4s2, out_b2p4, occupy_2, real_num_2, out_stride=8, flops=flops)
        # out = self.bn3(out)
        out, flops = fn_forward(self.bn3, out, occupy_3, flops=flops)
        out = self.relu(out)
        # out_b3p8 = self.block3(out)
        out_b3p8, flops = block_forward(self.block3, out, occupy_3, real_num_3, stride=8, flops=flops)

        # pixel_dist=16
        # out = self.conv4p8s2(out_b3p8)
        out, occupy_4, real_num_4, flops = conv_downsample(self.conv4p8s2, out_b3p8, occupy_3, real_num_3, out_stride=16, flops=flops)
        # out = self.bn4(out)
        out, flops = fn_forward(self.bn4, out, occupy_4, flops=flops)
        out = self.relu(out)
        # out = self.block4(out)
        out, flops = block_forward(self.block4, out, occupy_4, real_num_4, stride=16, flops=flops)

        # pixel_dist=8
        # out = self.convtr4p16s2(out)
        out, flops = conv_upample(self.convtr4p16s2, out, occupy_4, real_num_4, out_b3p8.C, occupy_3, out_stride=8, flops=flops)
        # out = self.bntr4(out)
        out, flops = fn_forward(self.bntr4, out, occupy_3, flops=flops)
        out = self.relu(out)

        # out = me.cat(out, out_b3p8)
        out._F = torch.cat([out.F, out_b3p8.F], dim=-1)
        # out = self.block5(out)
        out, flops = block_forward(self.block5, out, occupy_3, real_num_3, stride=8, flops=flops)

        # pixel_dist=4
        # out = self.convtr5p8s2(out)
        out, flops = conv_upample(self.convtr5p8s2, out, occupy_3, real_num_3, out_b2p4.C, occupy_2, out_stride=4, flops=flops)
        # out = self.bntr5(out)
        out, flops = fn_forward(self.bntr5, out, occupy_2, flops=flops)
        out = self.relu(out)

        # out = me.cat(out, out_b2p4)
        out._F = torch.cat([out.F, out_b2p4.F], dim=-1, flops=flops)
        # out = self.block6(out)
        out, flops = block_forward(self.block6, out, occupy_2, real_num_2, stride=4, flops=flops)

        # pixel_dist=2
        # out = self.convtr6p4s2(out)
        out, flops = conv_upample(self.convtr6p4s2, out, occupy_2, real_num_2, out_b1p2.C, occupy_1, out_stride=2, flops=flops)
        # out = self.bntr6(out)
        out, flops = fn_forward(self.bntr6, out, occupy_1, flops=flops)
        out = self.relu(out)

        # out = me.cat(out, out_b1p2)
        out._F = torch.cat([out.F, out_b1p2.F], dim=-1)
        # out = self.block7(out)
        out, flops = block_forward(self.block7, out, occupy_1, real_num_1, stride=2, flops=flops)

        # pixel_dist=1
        # out = self.convtr7p2s2(out)
        out, flops = conv_upample(self.convtr7p2s2, out, occupy_1, real_num_1, out_p1.C, occupy, out_stride=1, flops=flops)
        # out = self.bntr7(out)
        out, flops = fn_forward(self.bntr7, out, occupy, flops=flops)
        out = self.relu(out)

        # out = me.cat(out, out_p1)
        out._F = torch.cat([out.F, out_p1.F], dim=-1)

        residual = out
        # x = block[i].conv1(x)
        out, flops = conv_nosample(self.block8[0].conv1, out, occupy, real_num, flops=flops)
        # x = block[i].norm1(x)
        out, flops = fn_forward(self.block8[0].norm1, out, occupy, flops=flops)
        out = self.block8[0].relu(out)

        return [occupy, real_num, residual, out, flops]


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
        [occupy, real_num, residual, out, flops] = interm
        occupy, residual, out = \
            occupy.to('cuda'), \
            ME.SparseTensor(residual.F, residual.C, tensor_stride=residual.tensor_stride).to('cuda'), \
            ME.SparseTensor(out.F, out.C, tensor_stride=out.tensor_stride).to('cuda')

        # x = block[i].conv2(x)
        out, flops = conv_nosample(self.block8[0].conv2, out, occupy, real_num, flops=flops)
        # x = block[i].norm2(x)
        out, flops = fn_forward(self.block8[0].norm2, out, occupy, flops=flops)
        if self.block8[0].downsample is not None:
            # residual = block[i].downsample[0](residual)
            residual, flops = conv_nosample(self.block8[0].downsample[0], residual, occupy, real_num, flops=flops)
            # residual = block[i].downsample[1](residual)
            residual, flops = fn_forward(self.block8[0].downsample[1], residual, occupy, flops=flops)
        out._F = out.F + residual.F
        out = self.block8[0].relu(out)
        
        # out = self.block8(out)
        out, flops = block_forward([self.block8[1]], out, occupy, real_num, flops=flops)
        
        # out = self.final(out)
        out, flops = conv_nosample(self.final, out, occupy, real_num, flops=flops, mul_occupy_after=False)
        return out, flops


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
        out, flops = conv_nosample(self.conv0p1s1, x, occupy, real_num)
        # out = self.bn0(out)
        out, flops = fn_forward(self.bn0, out, occupy, flops=flops)
        out_p1 = self.relu(out)

        # out = self.conv1p1s2(out_p1)
        out, occupy_1, real_num_1, flops = conv_downsample(self.conv1p1s2, out_p1, occupy, real_num, out_stride=2, flops=flops)
        # out = self.bn1(out)
        out, flops = fn_forward(self.bn1, out, occupy_1, flops=flops)
        out = self.relu(out)
        # out_b1p2 = self.block1(out)
        out_b1p2, flops = block_forward(self.block1, out, occupy_1, real_num_1, stride=2, flops=flops)

        # out = self.conv2p2s2(out_b1p2)
        out, occupy_2, real_num_2, flops = conv_downsample(self.conv2p2s2, out_b1p2, occupy_1, real_num_1, out_stride=4, flops=flops)
        # out = self.bn2(out)
        out, flops = fn_forward(self.bn2, out, occupy_2, flops=flops)
        out = self.relu(out)
        # out_b2p4 = self.block2(out)
        out_b2p4, flops = block_forward(self.block2, out, occupy_2, real_num_2, stride=4, flops=flops)

        # out = self.conv3p4s2(out_b2p4)
        out, occupy_3, real_num_3, flops = conv_downsample(self.conv3p4s2, out_b2p4, occupy_2, real_num_2, out_stride=8, flops=flops)
        # out = self.bn3(out)
        out, flops = fn_forward(self.bn3, out, occupy_3, flops=flops)
        out = self.relu(out)
        # out_b3p8 = self.block3(out)
        out_b3p8, flops = block_forward(self.block3, out, occupy_3, real_num_3, stride=8, flops=flops)

        # pixel_dist=16
        # out = self.conv4p8s2(out_b3p8)
        out, occupy_4, real_num_4, flops = conv_downsample(self.conv4p8s2, out_b3p8, occupy_3, real_num_3, out_stride=16, flops=flops)
        # out = self.bn4(out)
        out, flops = fn_forward(self.bn4, out, occupy_4, flops=flops)
        out = self.relu(out)
        # out = self.block4(out)
        out, flops = block_forward(self.block4, out, occupy_4, real_num_4, stride=16, flops=flops)

        # pixel_dist=8
        # out = self.convtr4p16s2(out)
        out, flops = conv_upample(self.convtr4p16s2, out, occupy_4, real_num_4, out_b3p8.C, occupy_3, out_stride=8, flops=flops)
        # out = self.bntr4(out)
        out, flops = fn_forward(self.bntr4, out, occupy_3, flops=flops)
        out = self.relu(out)

        # out = me.cat(out, out_b3p8)
        out._F = torch.cat([out.F, out_b3p8.F], dim=-1)
        # out = self.block5(out)
        out, flops = block_forward(self.block5, out, occupy_3, real_num_3, stride=8, flops=flops)

        # pixel_dist=4
        # out = self.convtr5p8s2(out)
        out, flops = conv_upample(self.convtr5p8s2, out, occupy_3, real_num_3, out_b2p4.C, occupy_2, out_stride=4, flops=flops)
        # out = self.bntr5(out)
        out, flops = fn_forward(self.bntr5, out, occupy_2, flops=flops)
        out = self.relu(out)

        # out = me.cat(out, out_b2p4)
        out._F = torch.cat([out.F, out_b2p4.F], dim=-1)
        # out = self.block6(out)
        out, flops = block_forward(self.block6, out, occupy_2, real_num_2, stride=4, flops=flops)

        # pixel_dist=2
        # out = self.convtr6p4s2(out)
        out, flops = conv_upample(self.convtr6p4s2, out, occupy_2, real_num_2, out_b1p2.C, occupy_1, out_stride=2, flops=flops)
        # out = self.bntr6(out)
        out, flops = fn_forward(self.bntr6, out, occupy_1, flops=flops)
        out = self.relu(out)

        # out = me.cat(out, out_b1p2)
        out._F = torch.cat([out.F, out_b1p2.F], dim=-1)
        # out = self.block7(out)
        out, flops = block_forward([self.block7[0]], out, occupy_1, real_num_1, stride=2, flops=flops)

        return [[occupy, real_num, out_p1], [occupy_1, real_num_1, out], flops]


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
        [[occupy, real_num, out_p1], [occupy_1, real_num_1, out], flops] = interm
        occupy, out_p1, occupy_1, out = \
            occupy.to('cuda'), ME.SparseTensor(out_p1.F, out_p1.C, tensor_stride=out_p1.tensor_stride).to('cuda'), \
            occupy_1.to('cuda'), ME.SparseTensor(out.F, out.C, tensor_stride=out.tensor_stride).to('cuda')

        out, flops = block_forward([self.block7[1]], out, occupy_1, real_num_1, stride=2, flops=flops)

        # pixel_dist=1
        # out = self.convtr7p2s2(out)
        out, flops = conv_upample(self.convtr7p2s2, out, occupy_1, real_num_1, out_p1.C, occupy, out_stride=1, flops=flops)
        # out = self.bntr7(out)
        out, flops = fn_forward(self.bntr7, out, occupy, flops=flops)
        out = self.relu(out)

        # out = me.cat(out, out_p1)
        out._F = torch.cat([out.F, out_p1.F], dim=-1)

        residual = out
        # x = block[i].conv1(x)
        out, flops = conv_nosample(self.block8[0].conv1, out, occupy, real_num, flops=flops)
        # x = block[i].norm1(x)
        out, flops = fn_forward(self.block8[0].norm1, out, occupy, flops=flops)
        out = self.block8[0].relu(out)

        return [occupy, real_num, residual, out, flops]


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
        [occupy, real_num, residual, out, flops] = interm
        occupy, residual, out = \
            occupy.to('cuda'), ME.SparseTensor(residual.F, residual.C, tensor_stride=residual.tensor_stride).to('cuda'), \
            ME.SparseTensor(out.F, out.C, tensor_stride=out.tensor_stride).to('cuda')
        
        # out = self.block8(out)
        # x = block[i].conv2(x)
        out, flops = conv_nosample(self.block8[0].conv2, out, occupy, real_num, flops=flops)
        # x = block[i].norm2(x)
        out, flops = fn_forward(self.block8[0].norm2, out, occupy, flops=flops)
        if self.block8[0].downsample is not None:
            # residual = block[i].downsample[0](residual)
            residual, flops = conv_nosample(self.block8[0].downsample[0], residual, occupy, real_num, flops=flops)
            # residual = block[i].downsample[1](residual)
            residual, flops = fn_forward(self.block8[0].downsample[1], residual, occupy, flops=flops)
        out._F = out.F + residual.F
        out = self.block8[0].relu(out)

        return [occupy, real_num, out, flops]


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
        [occupy, real_num, out, flops] = interm
        occupy, out = \
            occupy.to('cuda'), ME.SparseTensor(out.F, out.C).to('cuda')

        out, flops = block_forward([self.block8[1]], out, occupy, real_num, flops=flops)
        
        # out = self.final(out)
        out, flops = conv_nosample(self.final, out, occupy, real_num, flops=flops, mul_occupy_after=False)
        return out, flops


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
        out, flops = conv_nosample(self.conv0p1s1, x, occupy, real_num)
        # out = self.bn0(out)
        out, flops = fn_forward(self.bn0, out, occupy, flops=flops)
        out_p1 = self.relu(out)

        # out = self.conv1p1s2(out_p1)
        out, occupy_1, real_num_1, flops = conv_downsample(self.conv1p1s2, out_p1, occupy, real_num, out_stride=2, flops=flops)
        # out = self.bn1(out)
        out, flops = fn_forward(self.bn1, out, occupy_1, flops=flops)
        out = self.relu(out)
        # out_b1p2 = self.block1(out)
        out_b1p2, flops = block_forward(self.block1, out, occupy_1, real_num_1, stride=2, flops=flops)

        # out = self.conv2p2s2(out_b1p2)
        out, occupy_2, real_num_2, flops = conv_downsample(self.conv2p2s2, out_b1p2, occupy_1, real_num_1, out_stride=4, flops=flops)
        # out = self.bn2(out)
        out, flops = fn_forward(self.bn2, out, occupy_2, flops=flops)
        out = self.relu(out)
        # out_b2p4 = self.block2(out)
        out_b2p4, flops = block_forward(self.block2, out, occupy_2, real_num_2, stride=4, flops=flops)

        # out = self.conv3p4s2(out_b2p4)
        out, occupy_3, real_num_3, flops = conv_downsample(self.conv3p4s2, out_b2p4, occupy_2, real_num_2, out_stride=8, flops=flops)
        # out = self.bn3(out)
        out, flops = fn_forward(self.bn3, out, occupy_3, flops=flops)
        out = self.relu(out)
        # out_b3p8 = self.block3(out)
        out_b3p8, flops = block_forward(self.block3, out, occupy_3, real_num_3, stride=8, flops=flops)

        # pixel_dist=16
        # out = self.conv4p8s2(out_b3p8)
        out, occupy_4, real_num_4, flops = conv_downsample(self.conv4p8s2, out_b3p8, occupy_3, real_num_3, out_stride=16, flops=flops)
        # out = self.bn4(out)
        out, flops = fn_forward(self.bn4, out, occupy_4, flops=flops)
        out = self.relu(out)
        # out = self.block4(out)
        out, flops = block_forward(self.block4, out, occupy_4, real_num_4, stride=16, flops=flops)

        return [[occupy, real_num, out_p1], [occupy_1, real_num_1, out_b1p2], [occupy_2, real_num_2, out_b2p4], \
            [occupy_3, real_num_3, out_b3p8], [occupy_4, real_num_4, out], flops]


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
            [occupy_3, real_num_3, out_b3p8], [occupy_4, real_num_4, out], flops] = interm
        occupy, out_p1, occupy_1, out_b1p2, occupy_2, out_b2p4, occupy_3, out_b3p8, occupy_4, out = \
            occupy.to('cuda'), ME.SparseTensor(out_p1.F, out_p1.C, tensor_stride=out_p1.tensor_stride).to('cuda'), \
            occupy_1.to('cuda'), ME.SparseTensor(out_b1p2.F, out_b1p2.C, tensor_stride=out_b1p2.tensor_stride).to('cuda'), \
            occupy_2.to('cuda'), ME.SparseTensor(out_b2p4.F, out_b2p4.C, tensor_stride=out_b2p4.tensor_stride).to('cuda'), \
            occupy_3.to('cuda'), ME.SparseTensor(out_b3p8.F, out_b3p8.C, tensor_stride=out_b3p8.tensor_stride).to('cuda'), \
            occupy_4.to('cuda'), ME.SparseTensor(out.F, out.C, tensor_stride=out.tensor_stride).to('cuda')

        # pixel_dist=8
        # out = self.convtr4p16s2(out)
        out, flops = conv_upample(self.convtr4p16s2, out, occupy_4, real_num_4, out_b3p8.C, occupy_3, out_stride=8, flops=flops)
        # out = self.bntr4(out)
        out, flops = fn_forward(self.bntr4, out, occupy_3, flops=flops)
        out = self.relu(out)

        # out = me.cat(out, out_b3p8)
        out._F = torch.cat([out.F, out_b3p8.F], dim=-1)
        # out = self.block5(out)
        out, flops = block_forward(self.block5, out, occupy_3, real_num_3, stride=8, flops=flops)

        # pixel_dist=4
        # out = self.convtr5p8s2(out)
        out, flops = conv_upample(self.convtr5p8s2, out, occupy_3, real_num_3, out_b2p4.C, occupy_2, out_stride=4, flops=flops)
        # out = self.bntr5(out)
        out, flops = fn_forward(self.bntr5, out, occupy_2, flops=flops)
        out = self.relu(out)

        # out = me.cat(out, out_b2p4)
        out._F = torch.cat([out.F, out_b2p4.F], dim=-1)
        # out = self.block6(out)
        out, flops = block_forward(self.block6, out, occupy_2, real_num_2, stride=4, flops=flops)

        # pixel_dist=2
        # out = self.convtr6p4s2(out)
        out, flops = conv_upample(self.convtr6p4s2, out, occupy_2, real_num_2, out_b1p2.C, occupy_1, out_stride=2, flops=flops)
        # out = self.bntr6(out)
        out, flops = fn_forward(self.bntr6, out, occupy_1, flops=flops)
        out = self.relu(out)

        # out = me.cat(out, out_b1p2)
        out._F = torch.cat([out.F, out_b1p2.F], dim=-1)
        # out = self.block7(out)
        out, flops = block_forward([self.block7[0]], out, occupy_1, real_num_1, stride=2, flops=flops)

        return [[occupy, real_num, out_p1], [occupy_1, real_num_1, out], flops]


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
        [[occupy, real_num, out_p1], [occupy_1, real_num_1, out], flops] = interm
        occupy, out_p1, occupy_1, out = \
            occupy.to('cuda'), ME.SparseTensor(out_p1.F, out_p1.C, tensor_stride=out_p1.tensor_stride).to('cuda'), \
            occupy_1.to('cuda'), ME.SparseTensor(out.F, out.C, tensor_stride=out.tensor_stride).to('cuda')

        out, flops = block_forward([self.block7[1]], out, occupy_1, real_num_1, stride=2, flops=flops)

        # pixel_dist=1
        # out = self.convtr7p2s2(out)
        out, flops = conv_upample(self.convtr7p2s2, out, occupy_1, real_num_1, out_p1.C, occupy, out_stride=1, flops=flops)
        # out = self.bntr7(out)
        out, flops = fn_forward(self.bntr7, out, occupy, flops=flops)
        out = self.relu(out)

        # out = me.cat(out, out_p1)
        out._F = torch.cat([out.F, out_p1.F], dim=-1)

        return [occupy, real_num, out, flops]


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
        [occupy, real_num, out, flops] = interm
        occupy, out = \
            occupy.to('cuda'), ME.SparseTensor(out.F, out.C, tensor_stride=out.tensor_stride).to('cuda')

        residual = out
        # x = block[i].conv1(x)
        out, flops = conv_nosample(self.block8[0].conv1, out, occupy, real_num, flops=flops)
        # x = block[i].norm1(x)
        out, flops = fn_forward(self.block8[0].norm1, out, occupy, flops=flops)
        out = self.block8[0].relu(out)

        return [occupy, real_num, residual, out, flops]


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
        [occupy, real_num, residual, out, flops] = interm
        occupy, residual, out = \
            occupy.to('cuda'), ME.SparseTensor(residual.F, residual.C, tensor_stride=residual.tensor_stride).to('cuda'), \
            ME.SparseTensor(out.F, out.C, tensor_stride=out.tensor_stride).to('cuda')
        
        # out = self.block8(out)
        # x = block[i].conv2(x)
        out, flops = conv_nosample(self.block8[0].conv2, out, occupy, real_num, flops=flops)
        # x = block[i].norm2(x)
        out, flops = fn_forward(self.block8[0].norm2, out, occupy, flops=flops)

        return [occupy, real_num, residual, out, flops]


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
        [occupy, real_num, residual, out, flops] = interm
        occupy, residual, out = \
            occupy.to('cuda'), ME.SparseTensor(residual.F, residual.C, tensor_stride=residual.tensor_stride).to('cuda'), \
            ME.SparseTensor(out.F, out.C, tensor_stride=out.tensor_stride).to('cuda')
        
        if self.block8[0].downsample is not None:
            # residual = block[i].downsample[0](residual)
            residual, flops = conv_nosample(self.block8[0].downsample[0], residual, occupy, real_num, flops=flops)
            # residual = block[i].downsample[1](residual)
            residual, flops = fn_forward(self.block8[0].downsample[1], residual, occupy, flops=flops)
        out._F = out.F + residual.F
        out = self.block8[0].relu(out)

        return [occupy, real_num, out, flops]


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
        [occupy, real_num, out, flops] = interm
        occupy, out = \
            occupy.to('cuda'), ME.SparseTensor(out.F, out.C).to('cuda')

        out, flops = block_forward([self.block8[1]], out, occupy, real_num, flops=flops)
        
        return [occupy, real_num, out, flops]


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
        [occupy, real_num, out, flops] = interm
        occupy, out = \
            occupy.to('cuda'), ME.SparseTensor(out.F, out.C).to('cuda')
        
        # out = self.final(out)
        out, flops = conv_nosample(self.final, out, occupy, real_num, flops=flops, mul_occupy_after=False)
        return out, flops

