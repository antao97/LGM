import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import dynconv.dynconv as dyn
import dynconv.models.resnet_util


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    """Standard residual block """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, sparse=False, dynamics_aware=False, lamb=10, diff_type='gg'):
        super(BasicBlock, self).__init__()
        assert groups == 1
        assert dilation == 1
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.sparse = sparse
        self.dynamics_aware = dynamics_aware
        self.lamb = lamb
        self.diff_type = diff_type

        if sparse:
            # in the resnet basic block, the first convolution is already strided, so mask_stride = 1
            self.masker = dyn.MaskUnit(channels=inplanes, stride=stride, dilate_stride=1, 
                                    dynamics_aware=self.dynamics_aware, lamb=self.lamb, diff_type=self.diff_type)

        self.fast = False

    def forward(self, input):
        x, meta = input
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        if not self.sparse:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out += identity
        else:
            assert meta is not None
            m = self.masker(x, meta)
            mask_dilate, mask = m['dilate'], m['std']

            x = dyn.conv3x3(self.conv1, x, None, mask_dilate)
            x = dyn.bn_relu(self.bn1, self.relu, x, mask_dilate)
            x = dyn.conv3x3(self.conv2, x, mask_dilate, mask)
            x = dyn.bn_relu(self.bn2, None, x, mask)
            out = identity + dyn.apply_mask(x, mask)

        out = self.relu(out)
        return out, meta

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, sparse=False, dynamics_aware=False, lamb=10, diff_type='gg'):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        print(f'Bottleneck - sparse: {sparse}: inp {inplanes}, hidden_dim {width}, ' + 
              f'oup {planes * self.expansion}, stride {stride}')

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.sparse = sparse
        self.fast = True
        self.dynamics_aware = dynamics_aware
        self.lamb = lamb
        self.diff_type = diff_type

        if sparse:
            self.masker = dyn.MaskUnit(channels=inplanes, stride=stride, dilate_stride=stride, 
                                    dynamics_aware=self.dynamics_aware, lamb=self.lamb, diff_type=self.diff_type)

    def forward(self, input):
        x, meta = input
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        if not self.sparse:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)
            out += identity
        else:
            assert meta is not None
            m = self.masker(x, meta)
            mask_dilate, mask = m['dilate'], m['std']

            x = dyn.conv1x1(self.conv1, x, mask_dilate)
            x = dyn.bn_relu(self.bn1, self.relu, x, mask_dilate)
            x = dyn.conv3x3(self.conv2, x, mask_dilate, mask)
            x = dyn.bn_relu(self.bn2, self.relu, x, mask)
            x = dyn.conv1x1(self.conv3, x, mask)
            x = dyn.bn_relu(self.bn3, None, x, mask)
            out = identity + dyn.apply_mask(x, mask)

        out = self.relu(out)
        return out, meta
