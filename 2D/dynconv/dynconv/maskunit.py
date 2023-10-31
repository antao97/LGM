import torch
import torch.nn as nn
import torch.nn.functional as F
from dynconv.utils import logger


class Mask():
    '''
    Class that holds the mask properties

    hard: the hard/binary mask (1 or 0), 4-dim tensor
    soft (optional): the float mask, same shape as hard
    active_positions: the amount of positions where hard == 1
    total_positions: the total amount of positions 
                        (typically batch_size * output_width * output_height)
    '''
    def __init__(self, hard, soft=None):
        assert hard.dim() == 4
        assert hard.shape[1] == 1
        assert soft is None or soft.shape == hard.shape

        self.hard = hard
        self.active_positions = torch.sum(hard) # this must be kept backpropagatable!
        self.total_positions = hard.numel()
        self.soft = soft
        
        self.flops_per_position = 0
    
    def size(self):
        return self.hard.shape

    def __repr__(self):
        return f'Mask with {self.active_positions}/{self.total_positions} positions, and {self.flops_per_position} accumulated FLOPS per position'

class MaskUnit(nn.Module):
    ''' 
    Generates the mask and applies the gumbel softmax trick 
    '''

    def __init__(self, channels, stride=1, dilate_stride=1, dynamics_aware=False, lamb=10, diff_type='gg'):
        super(MaskUnit, self).__init__()
        self.maskconv = Squeeze(channels=channels, stride=stride)
        self.gumbel = Gumbel(dynamics_aware=dynamics_aware, lamb=lamb, diff_type=diff_type)
        self.expandmask = ExpandMask(stride=dilate_stride, dynamics_aware=dynamics_aware, lamb=lamb, diff_type=diff_type)

    def forward(self, x, meta):
        soft = self.maskconv(x)
        hard = self.gumbel(soft, meta['gumbel_temp'], meta['gumbel_noise'])
        mask = Mask(hard, soft)
        
        hard_dilate = self.expandmask(mask.hard)
        mask_dilate = Mask(hard_dilate)
        
        m = {'std': mask, 'dilate': mask_dilate}
        meta['masks'].append(m)
        return m


## Gumbel

class Gumbel(nn.Module):
    ''' 
    Returns differentiable discrete outputs. Applies a Gumbel-Softmax trick on every element of x. 
    '''
    def __init__(self, eps=1e-8, dynamics_aware=False, lamb=None, diff_type='gg'):
        super(Gumbel, self).__init__()
        self.eps = eps
        self.dynamics_aware = dynamics_aware
        self.lamb = lamb
        self.diff_type = diff_type

    def forward(self, x, gumbel_temp=1.0, gumbel_noise=True):
        if self.dynamics_aware:
            soft = torch.sigmoid(self.lamb * x)
            if self.diff_type == 'gg':
                return soft
            elif self.diff_type == 'fg':
                hard = ((soft >= 0.5).float() - soft).detach() + soft
                return hard

        if not self.training:  # no Gumbel noise during inference
            return (x >= 0).float()

        logger.add('gumbel_noise', gumbel_noise)
        logger.add('gumbel_temp', gumbel_temp)

        if gumbel_noise:
            eps = self.eps
            U1, U2 = torch.rand_like(x), torch.rand_like(x)
            g1, g2 = -torch.log(-torch.log(U1 + eps)+eps), - \
                torch.log(-torch.log(U2 + eps)+eps)
            x = x + g1 - g2

        soft = torch.sigmoid(x / gumbel_temp)
        hard = ((soft >= 0.5).float() - soft).detach() + soft
        assert not torch.any(torch.isnan(hard))
        return hard

## Mask convs
class Squeeze(nn.Module):
    """ 
    Squeeze module to predict masks 
    """

    def __init__(self, channels, stride=1):
        super(Squeeze, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 1, bias=True)
        self.conv = nn.Conv2d(channels, 1, stride=stride,
                              kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, 1, 1, 1)
        z = self.conv(x)
        return z + y.expand_as(z)

class ExpandMask(nn.Module):
    def __init__(self, stride, padding=1, dynamics_aware=False, lamb=None, diff_type='gg'): 
        super(ExpandMask, self).__init__()
        self.stride=stride
        self.padding = padding
        self.dynamics_aware = dynamics_aware
        self.lamb = lamb
        self.diff_type = diff_type

    def forward(self, x):
        assert x.shape[1] == 1

        if self.stride > 1:
            self.pad_kernel = torch.zeros( (1,1,self.stride, self.stride), device=x.device)
            self.pad_kernel[0,0,0,0] = 1
        self.dilate_kernel = torch.ones((1,1,1+2*self.padding,1+2*self.padding), device=x.device)

        x = x.float()
        if self.stride > 1:
            x = F.conv_transpose2d(x, self.pad_kernel, stride=self.stride, groups=x.size(1))
        x = F.conv2d(x, self.dilate_kernel, padding=self.padding, stride=1)
        if self.dynamics_aware:
            soft = torch.sigmoid(self.lamb*(x-0.5))
            if self.diff_type == 'gg':
                return soft
            elif self.diff_type == 'fg':
                hard = ((soft >= 0.5).float() - soft).detach() + soft
                return hard
        else:
            return x > 0.5
