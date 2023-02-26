# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
from mmcv.runner import load_checkpoint
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from torch.autograd import Function
import torch.nn as nn
from .. import build_detector
from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .single_stage import SingleStageDetector
from mmdet.utils import collect_env, get_root_logger
#        add loogers to debug

# logger = get_root_logger()
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch import einsum
from string import Template
from collections import namedtuple
import cupy

Stream = namedtuple('Stream', ['ptr'])

def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'

@cupy.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)

CUDA_NUM_THREADS = 1024

kernel_loop = '''
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)
'''

def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS

_aggregation_zeropad_forward_kernel = kernel_loop + '''
extern "C"
__global__ void aggregation_zeropad_forward_kernel(
const ${Dtype}* bottom_data, const ${Dtype}* weight_data, ${Dtype}* top_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${weight_heads} / ${input_channels} / ${top_height} / ${top_width};
    const int head = (index / ${top_width} / ${top_height} / ${input_channels}) % ${weight_heads};
    const int c = (index / ${top_width} / ${top_height}) % ${input_channels};
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};
    ${Dtype} value = 0;
    for (int kh = 0; kh < ${kernel_h}; ++kh) {
      for (int kw = 0; kw < ${kernel_w}; ++kw) {
        const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
        const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
        if ((h_in >= 0) && (h_in < ${bottom_height}) && (w_in >= 0) && (w_in < ${bottom_width})) {
          const int offset_bottom = ((n * ${input_channels} + c) * ${bottom_height} + h_in) * ${bottom_width} + w_in;
          const int offset_weight = (((n * ${weight_heads} + head) * ${weight_channels} + c % ${weight_channels}) * ${kernel_h} * ${kernel_w} + (kh * ${kernel_w} + kw)) * ${top_height} * ${top_width} + h * ${top_width} + w;
          value += weight_data[offset_weight] * bottom_data[offset_bottom];
        }
      }
    }
    top_data[index] = value;
  }
}
'''

_aggregation_zeropad_input_backward_kernel = kernel_loop + '''
extern "C"
__global__ void aggregation_zeropad_input_backward_kernel(
    const ${Dtype}* const top_diff, const ${Dtype}* const weight_data, ${Dtype}* bottom_diff) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${input_channels} / ${bottom_height} / ${bottom_width};
    const int c = (index / ${bottom_height} / ${bottom_width}) % ${input_channels};
    const int h = (index / ${bottom_width}) % ${bottom_height};
    const int w = index % ${bottom_width};
    ${Dtype} value = 0;
    for (int head = 0; head < ${weight_heads}; ++head) {
        for (int kh = 0; kh < ${kernel_h}; ++kh) {
          for (int kw = 0; kw < ${kernel_w}; ++kw) {
            const int h_out_s = h + ${pad_h} - kh * ${dilation_h};
            const int w_out_s = w + ${pad_w} - kw * ${dilation_w};
            if (((h_out_s % ${stride_h}) == 0) && ((w_out_s % ${stride_w}) == 0)) {
              const int h_out = h_out_s / ${stride_h};
              const int w_out = w_out_s / ${stride_w};
              if ((h_out >= 0) && (h_out < ${top_height}) && (w_out >= 0) && (w_out < ${top_width})) {
                const int offset_top = (((n * ${weight_heads} + head) * ${input_channels} + c) * ${top_height} + h_out) * ${top_width} + w_out;
                const int offset_weight = (((n * ${weight_heads} + head) * ${weight_channels} + c % ${weight_channels}) * ${kernel_h} * ${kernel_w} + (kh * ${kernel_w} + kw)) * ${top_height} * ${top_width} + h_out * ${top_width} + w_out;
                value += weight_data[offset_weight] * top_diff[offset_top];
              }
            }
          }
        }
    }
    bottom_diff[index] = value;
  }
}
'''

_aggregation_zeropad_weight_backward_kernel = kernel_loop + '''
extern "C"
__global__ void aggregation_zeropad_weight_backward_kernel(
    const ${Dtype}* const top_diff, const ${Dtype}* const bottom_data, ${Dtype}* weight_diff) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${weight_heads} / ${weight_channels} / ${top_height} / ${top_width};
    const int head = (index / ${top_width} / ${top_height} / ${weight_channels}) % ${weight_heads};
    const int c = (index / ${top_width} / ${top_height}) % ${weight_channels};
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};
    for (int kh = 0; kh < ${kernel_h}; ++kh) {
      for (int kw = 0; kw < ${kernel_w}; ++kw) {
        const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
        const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
        const int offset_weight = (((n * ${weight_heads} + head) * ${weight_channels} + c) * ${kernel_h} * ${kernel_w} + (kh * ${kernel_w} + kw)) * ${top_height} * ${top_width} + h * ${top_width} + w;
        ${Dtype} value = 0;
        if ((h_in >= 0) && (h_in < ${bottom_height}) && (w_in >= 0) && (w_in < ${bottom_width})) {
          for (int cc = c; cc < ${input_channels}; cc += ${weight_channels}) {
            const int offset_bottom = ((n * ${input_channels} + cc) * ${bottom_height} + h_in) * ${bottom_width} + w_in;
            const int offset_top = (((n * ${weight_heads} + head) * ${input_channels} + cc) * ${top_height} + h) * ${top_width} + w;
            value += bottom_data[offset_bottom] * top_diff[offset_top];
          }
        }
        weight_diff[offset_weight] = value;
      }
    }
  }
}
'''



class AggregationZeropad(Function):
    @staticmethod
    def forward(ctx, input, weight, kernel_size, stride, padding, dilation):
        kernel_size, stride, padding, dilation = _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation)
        ctx.kernel_size, ctx.stride, ctx.padding, ctx.dilation = kernel_size, stride, padding, dilation
        assert input.dim() == 4 and input.is_cuda and weight.is_cuda
        batch_size, input_channels, input_height, input_width = input.size()
        _, weight_heads, weight_channels, weight_kernels, weight_height, weight_width = weight.size()
        output_height = int((input_height + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1)
        output_width = int((input_width + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1)
        assert output_height * output_width == weight_height * weight_width
        output = input.new(batch_size, weight_heads * input_channels, output_height, output_width)
        n = output.numel()
        if not input.is_contiguous():
            input = input.detach().clone()
        if not weight.is_contiguous():
            weight = weight.detach().clone()

        with torch.cuda.device_of(input):
            f = load_kernel('aggregation_zeropad_forward_kernel', _aggregation_zeropad_forward_kernel, Dtype=Dtype(input), nthreads=n,
                            num=batch_size, input_channels=input_channels, 
                            weight_heads=weight_heads, weight_channels=weight_channels,
                            bottom_height=input_height, bottom_width=input_width,
                            top_height=output_height, top_width=output_width,
                            kernel_h=kernel_size[0], kernel_w=kernel_size[1],
                            stride_h=stride[0], stride_w=stride[1],
                            dilation_h=dilation[0], dilation_w=dilation[1],
                            pad_h=padding[0], pad_w=padding[1])
            f(block=(CUDA_NUM_THREADS, 1, 1),
              grid=(GET_BLOCKS(n), 1, 1),
              args=[input.data_ptr(), weight.data_ptr(), output.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        ctx.save_for_backward(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel_size, stride, padding, dilation = ctx.kernel_size, ctx.stride, ctx.padding, ctx.dilation
        input, weight = ctx.saved_tensors
        assert grad_output.is_cuda
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        batch_size, input_channels, input_height, input_width = input.size()
        _, weight_heads, weight_channels, weight_kernels, weight_height, weight_width = weight.size()
        output_height, output_width = grad_output.size()[2:]
        grad_input, grad_weight = None, None
        opt = dict(Dtype=Dtype(grad_output),
                   num=batch_size, input_channels=input_channels, 
                   weight_heads=weight_heads, weight_channels=weight_channels,
                   bottom_height=input_height, bottom_width=input_width,
                   top_height=output_height, top_width=output_width,
                   kernel_h=kernel_size[0], kernel_w=kernel_size[1],
                   stride_h=stride[0], stride_w=stride[1],
                   dilation_h=dilation[0], dilation_w=dilation[1],
                   pad_h=padding[0], pad_w=padding[1])
        with torch.cuda.device_of(input):
            if ctx.needs_input_grad[0]:
                grad_input = input.new(input.size())
                n = grad_input.numel()
                opt['nthreads'] = n
                f = load_kernel('aggregation_zeropad_input_backward_kernel', _aggregation_zeropad_input_backward_kernel, **opt)
                f(block=(CUDA_NUM_THREADS, 1, 1),
                  grid=(GET_BLOCKS(n), 1, 1),
                  args=[grad_output.data_ptr(), weight.data_ptr(), grad_input.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
            if ctx.needs_input_grad[1]:
                grad_weight = weight.new(weight.size())
                n = grad_weight.numel() // weight.shape[3]
                opt['nthreads'] = n
                f = load_kernel('aggregation_zeropad_weight_backward_kernel', _aggregation_zeropad_weight_backward_kernel, **opt)
                f(block=(CUDA_NUM_THREADS, 1, 1),
                  grid=(GET_BLOCKS(n), 1, 1),
                  args=[grad_output.data_ptr(), input.data_ptr(), grad_weight.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return grad_input, grad_weight, None, None, None, None

def aggregation_zeropad(input, weight, kernel_size=3, stride=1, padding=0, dilation=1):
    assert input.shape[0] == weight.shape[0] and (input.shape[1] % weight.shape[2] == 0)
    if input.is_cuda:
        out = AggregationZeropad.apply(input, weight, kernel_size, stride, padding, dilation)
    else:
        #raise NotImplementedError
        out = AggregationZeropad.apply(input.cuda(), weight.cuda(), kernel_size, stride, padding, dilation)
        torch.cuda.synchronize()
        out = out.cpu()
    return out

class LocalConvolution(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        pad_mode: int = 0,
    ):
        super(LocalConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pad_mode = pad_mode

    def forward(self, input, weight):
        #if self.pad_mode == 0:
        out = aggregation_zeropad(
            input, 
            weight, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation)
        #else:
        #  out = aggregation_refpad(
        #    input, 
        #    weight, 
        #    kernel_size=self.kernel_size, 
        #    stride=self.stride, 
        #    padding=self.padding, 
        #    dilation=self.dilation)  
        return out


def swish(x, inplace: bool = False):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    """
    return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


class Swish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return swish(x, self.inplace)

class CotLayer(nn.Module):
    def __init__(self, dim, kernel_size):
        super(CotLayer, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, self.kernel_size, stride=1, padding=self.kernel_size//2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv2d(2*dim, dim//factor, 1, bias=False),
            nn.BatchNorm2d(dim//factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//factor, pow(kernel_size, 2) * dim // share_planes, kernel_size=1),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=pow(kernel_size, 2) * dim // share_planes)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(dim)
        )

        self.local_conv = LocalConvolution(dim, dim, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2, dilation=1)
        self.bn = nn.BatchNorm2d(dim)
#        act = get_act_layer('swish')
        self.act = nn.SiLU(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv2d(dim, attn_chs, 1),
            nn.BatchNorm2d(attn_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(attn_chs, self.radix*dim, 1)
        )

    def forward(self, x):
        k = self.key_embed(x)
        qk = torch.cat([x, k], dim=1)
        b, c, qk_hh, qk_ww = qk.size()

        w = self.embed(qk)
        w = w.view(b, 1, -1, self.kernel_size*self.kernel_size, qk_hh, qk_ww)
        
        x = self.conv1x1(x)
        x = self.local_conv(x, w)
        x = self.bn(x)
        x = self.act(x)

        B, C, H, W = x.shape
        x = x.view(B, C, 1, H, W)
        k = k.view(B, C, 1, H, W)
        x = torch.cat([x, k], dim=2)

        x_gap = x.sum(dim=2)
        x_gap = x_gap.mean((2, 3), keepdim=True)
        x_attn = self.se(x_gap)
        x_attn = x_attn.view(B, C, self.radix)
        x_attn = F.softmax(x_attn, dim=2)
        out = (x * x_attn.reshape((B, C, self.radix, 1, 1))).sum(dim=2)
        
        return out.contiguous()

def dist2(tensor_a, tensor_b,
    fg_mask=None,
    attention_mask=None, channel_attention_mask=None
    ):
    diff = (tensor_a - tensor_b) ** 2
    #   print(diff.size())      batchsize x 1 x W x H,
    #   print(attention_mask.size()) batchsize x 1 x W x H
    diff = diff * attention_mask
    diff = diff * channel_attention_mask
    diff = diff * fg_mask
    diff = torch.sum(diff) ** 0.5
    return diff


def calculate_gaussian(old_w, old_h):
    w = 2*old_w
    h = 2*old_h
    cx = w / 2
    cy = h / 2
    if cx == 0:
        cx += 1
    if cy == 0:
        cy += 1
    x0 = cx.repeat(1, w)
    y0 = cy.repeat(h, 1)
    x = torch.arange(w).cuda()
    y = torch.unsqueeze(torch.arange(h), dim=1).cuda()
    gaussian_mask = torch.exp(-0.5*((x-x0)/cx)**2) * torch.exp(-0.5*((y-y0)/cy)**2)
    bbox = torch.ones((old_h,old_w)).cuda()
    gap_w = torch.floor((w-old_w)/2).int()
    gap_h = torch.floor((h-old_h)/2).int()
    gaussian_mask[gap_h:gap_h+old_h, gap_w:gap_w+old_w] = bbox
    gaussian_mask = 1/w/h*gaussian_mask.double().cuda()
    return gaussian_mask

class NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True, downsample_stride=2):
        super(NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(downsample_stride, downsample_stride))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :
        :
        '''

        batch_size = x.size(0)  #   2 , 256 , 300 , 300

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)   #   2 , 128 , 150 x 150
        g_x = g_x.permute(0, 2, 1)                                  #   2 , 150 x 150, 128

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)   #   2 , 128 , 300 x 300
        theta_x = theta_x.permute(0, 2, 1)                                  #   2 , 300 x 300 , 128
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)       #   2 , 128 , 150 x 150
        f = torch.matmul(theta_x, phi_x)    #   2 , 300x300 , 150x150
        N = f.size(-1)  #   150 x 150
        f_div_C = f / N #   2 , 300x300, 150x150

        y = torch.matmul(f_div_C, g_x)  #   2, 300x300, 128
        y = y.permute(0, 2, 1).contiguous() #   2, 128, 300x300
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


@DETECTORS.register_module()
class CotKnowledgeDistillationSingleStageDetector(SingleStageDetector):
    r"""Implementation of `Distilling the Knowledge in a Neural Network.
    <https://arxiv.org/abs/1503.02531>`_.
    Args:
        teacher_config (str | dict): Config file path
            or the config object of teacher model.
        teacher_ckpt (str, optional): Checkpoint path of teacher model.
            If left as None, the model will not load any weights.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 teacher_config,
                 teacher_ckpt=None,
                 eval_teacher=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                         pretrained)
        self.eval_teacher = eval_teacher
        # Build teacher model
        if isinstance(teacher_config, str):
            teacher_config = mmcv.Config.fromfile(teacher_config)
        self.teacher_model = build_detector(teacher_config['model'])
        if teacher_ckpt is not None:
            load_checkpoint(
                self.teacher_model, teacher_ckpt, map_location='cpu')
        ## add adaptation layers
        self.adaptation_type = '1x1conv'
        # self.bbox_feat_adaptation = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        #   self.cls_adaptation = nn.Linear(1024, 1024)
        #   self.reg_adaptation = nn.Linear(1024, 1024)
        self.channel_wise_adaptation = nn.ModuleList([
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256)
        ])

        self.spatial_wise_adaptation = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
        ])

        #   self.roi_adaptation_layer = nn.Conv2d(256, 256, kernel_size=1)
        if self.adaptation_type == '3x3conv':
            #   3x3 conv
            self.adaptation_layers = nn.ModuleList([
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            ])
        if self.adaptation_type == '1x1conv':
            #   1x1 conv
            self.adaptation_layers = nn.ModuleList([
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            ])

        if self.adaptation_type == '3x3conv+bn':
            #   3x3 conv + bn
            self.adaptation_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256, affine=True)
                )
            ])
        if self.adaptation_type == '1x1conv+bn':
            #   1x1 conv + bn
            self.adaptation_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(256, affine=True)
                )
            ])
        self.student_non_local = nn.ModuleList(
            [
                CotLayer(256, 1),
                CotLayer(256, 1),
                CotLayer(256, 1),
                CotLayer(256, 1),
                CotLayer(256, 1)
            ]
        )
        self.teacher_non_local = nn.ModuleList(
            [
                CotLayer(256, 1),
                CotLayer(256, 1),
                CotLayer(256, 1),
                CotLayer(256, 1),
                CotLayer(256, 1)
            ]
        )
        self.non_local_adaptation = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        ])
        self.deformable_adaptation = nn.ModuleList([
            ConvModule(256,256,3,stride=1,padding=1,
            conv_cfg = dict(type='DCN',deform_groups=1),
            norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)),
            ConvModule(256,256,3,stride=1,padding=1,
            conv_cfg = dict(type='DCN',deform_groups=1),
            norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)),
            ConvModule(256,256,3,stride=1,padding=1,
            conv_cfg = dict(type='DCN',deform_groups=1),
            norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)),
            ConvModule(256,256,3,stride=1,padding=1,
            conv_cfg = dict(type='DCN',deform_groups=1),
            norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)),
            ConvModule(256,256,3,stride=1,padding=1,
            conv_cfg = dict(type='DCN',deform_groups=1),
            norm_cfg = dict(type='GN', num_groups=32, requires_grad=True))])

        # self.student_non_local = nn.ModuleList(
        #     [
        #         NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=8),
        #         NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=4),
        #         NonLocalBlockND(in_channels=256),
        #         NonLocalBlockND(in_channels=256),
        #         NonLocalBlockND(in_channels=256)
        #     ]
        # )
        # self.teacher_non_local = nn.ModuleList(
        #     [
        #         NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=8),
        #         NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=4),
        #         NonLocalBlockND(in_channels=256),
        #         NonLocalBlockND(in_channels=256),
        #         NonLocalBlockND(in_channels=256)
        #     ]
        # )
        # self.non_local_adaptation = nn.ModuleList([
        #     nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
        #     nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
        #     nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
        #     nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
        #     nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        # ])

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(img)
        with torch.no_grad():
            teacher_x = self.teacher_model.extract_feat(img)
            losses_T = self.teacher_model.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)

        losses = self.bbox_head.forward_train(x, img_metas,
                                             gt_bboxes, gt_labels,
                                             gt_bboxes_ignore)


        # with torch.no_grad():
        #     t_feats = self.teacher_model.extract_feat(img)
        # for _i in range(len(t_feats)):
        #     N,C,H,W = x[_i].shape
        #     value = torch.abs(x[_i])
        #     fea_map = value.mean(axis=1, keepdim=True)
        #     S_attention = (H * W * F.softmax((fea_map/0.1).view(N,-1), dim=1)).view(N, H, W)
        #     Mask_fg = torch.zeros_like(S_attention)
        #     Mask_bg = torch.ones_like(S_attention)
        #     big_map = torch.zeros(N, H*2, W*2).cuda()
        #     wmin,wmax,hmin,hmax = [],[],[],[]
        #     for i in range(N):
        #         new_boxxes = torch.ones_like(gt_bboxes[i])
        #         new_boxxes[:, 0] = gt_bboxes[i][:, 0]/img_metas[i]['img_shape'][1]*W
        #         new_boxxes[:, 2] = gt_bboxes[i][:, 2]/img_metas[i]['img_shape'][1]*W
        #         new_boxxes[:, 1] = gt_bboxes[i][:, 1]/img_metas[i]['img_shape'][0]*H
        #         new_boxxes[:, 3] = gt_bboxes[i][:, 3]/img_metas[i]['img_shape'][0]*H
        #         wmin.append(torch.floor(new_boxxes[:, 0]).int())
        #         wmax.append(torch.ceil(new_boxxes[:, 2]).int())
        #         hmin.append(torch.floor(new_boxxes[:, 1]).int())
        #         hmax.append(torch.ceil(new_boxxes[:, 3]).int())
        #         for j in range(len(gt_bboxes[i])):
        #             w = (wmax[i][j]-wmin[i][j]).int()
        #             h = (hmax[i][j]-hmin[i][j]).int()
        #             big_map[i][2*hmin[i][j]:2*hmax[i][j], 2*wmin[i][j]:2*wmax[i][j]] = \
        #                     torch.maximum(big_map[i][2*hmin[i][j]:2*hmax[i][j], 2*wmin[i][j]:2*wmax[i][j]], calculate_gaussian(w, h))
        #         gap_W = np.floor((2*W-W)/2).astype(int)
        #         gap_H = np.floor((2*H-H)/2).astype(int)
        #         Mask_fg[i] = big_map[i][gap_H:gap_H+H, gap_W:gap_W+W]
        #         logger.info(f"Mask_fg: {Mask_fg[i]}")





        t = 0.1
        s_ratio = 1.0
        kd_feat_loss = 0
        kd_channel_loss = 0
        kd_spatial_loss = 0

        #   for channel attention
        c_t = 0.1
        c_s_ratio = 1.0

        with torch.no_grad():
            t_feats = self.teacher_model.extract_feat(img)
               
        if t_feats is not None:
            for _i in range(len(t_feats)):
                ## mask map
                N,C,H,W = x[_i].shape
                value = torch.abs(x[_i])
                fea_map = value.mean(axis=1, keepdim=True)
                S_attention = (H * W * F.softmax((fea_map/0.1).view(N,-1), dim=1)).view(N, H, W)
                Mask_fg = torch.zeros_like(S_attention)
                big_map = torch.zeros(N, H*2, W*2).cuda()
                wmin,wmax,hmin,hmax = [],[],[],[]
                for i in range(N):
                    new_boxxes = torch.ones_like(gt_bboxes[i])
                    new_boxxes[:, 0] = gt_bboxes[i][:, 0]/img_metas[i]['img_shape'][1]*W
                    new_boxxes[:, 2] = gt_bboxes[i][:, 2]/img_metas[i]['img_shape'][1]*W
                    new_boxxes[:, 1] = gt_bboxes[i][:, 1]/img_metas[i]['img_shape'][0]*H
                    new_boxxes[:, 3] = gt_bboxes[i][:, 3]/img_metas[i]['img_shape'][0]*H
                    wmin.append(torch.floor(new_boxxes[:, 0]).int())
                    wmax.append(torch.ceil(new_boxxes[:, 2]).int())
                    hmin.append(torch.floor(new_boxxes[:, 1]).int())
                    hmax.append(torch.ceil(new_boxxes[:, 3]).int())
                    for j in range(len(gt_bboxes[i])):
                        w = (wmax[i][j]-wmin[i][j]).int()
                        h = (hmax[i][j]-hmin[i][j]).int()
                        big_map[i][2*hmin[i][j]:2*hmax[i][j], 2*wmin[i][j]:2*wmax[i][j]] = \
                                torch.maximum(big_map[i][2*hmin[i][j]:2*hmax[i][j], 2*wmin[i][j]:2*wmax[i][j]], calculate_gaussian(w, h))
                    gap_W = np.floor((2*W-W)/2).astype(int)
                    gap_H = np.floor((2*H-H)/2).astype(int)
                    Mask_fg[i] = big_map[i][gap_H:gap_H+H, gap_W:gap_W+W]
                Mask_fg = Mask_fg.view(N,1,H,W)

                t_attention_mask = torch.mean(torch.abs(t_feats[_i]), [1], keepdim=True)
                size = t_attention_mask.size()
                t_attention_mask = t_attention_mask.view(x[0].size(0), -1)
                t_attention_mask = torch.softmax(t_attention_mask / t, dim=1) * size[-1] * size[-2]
                t_attention_mask = t_attention_mask.view(size)

                s_attention_mask = torch.mean(torch.abs(x[_i]), [1], keepdim=True)
                size = s_attention_mask.size()
                s_attention_mask = s_attention_mask.view(x[0].size(0), -1)
                s_attention_mask = torch.softmax(s_attention_mask / t, dim=1) * size[-1] * size[-2]
                s_attention_mask = s_attention_mask.view(size)

                c_t_attention_mask = torch.mean(torch.abs(t_feats[_i]), [2, 3], keepdim=True)  # 2 x 256 x 1 x1
                c_size = c_t_attention_mask.size()
                c_t_attention_mask = c_t_attention_mask.view(x[0].size(0), -1)  # 2 x 256
                c_t_attention_mask = torch.softmax(c_t_attention_mask / c_t, dim=1) * 256
                c_t_attention_mask = c_t_attention_mask.view(c_size)  # 2 x 256 -> 2 x 256 x 1 x 1

                c_s_attention_mask = torch.mean(torch.abs(x[_i]), [2, 3], keepdim=True)  # 2 x 256 x 1 x1
                c_size = c_s_attention_mask.size()
                c_s_attention_mask = c_s_attention_mask.view(x[0].size(0), -1)  # 2 x 256
                c_s_attention_mask = torch.softmax(c_s_attention_mask / c_t, dim=1) * 256
                c_s_attention_mask = c_s_attention_mask.view(c_size)  # 2 x 256 -> 2 x 256 x 1 x 1

                sum_attention_mask = (t_attention_mask + s_attention_mask * s_ratio) / (1 + s_ratio)
                sum_attention_mask = sum_attention_mask.detach()

                c_sum_attention_mask = (c_t_attention_mask + c_s_attention_mask * c_s_ratio) / (1 + c_s_ratio)
                c_sum_attention_mask = c_sum_attention_mask.detach()

                #t_deformable_mask  = self.deformable_adaptation[_i](t_feats[_i])
                # logger.info(f"teacher_deformable_mask is: {t_deformable_mask.size()}")
                # logger.info(f"teacher_deformable_mask is: {torch.count_nonzero(t_deformable_mask)}")

                #s_deformable_mask  = self.deformable_adaptation[_i](x[_i])
                #sum_deformable_mask = (t_deformable_mask + s_deformable_mask* c_s_ratio) / (1 + c_s_ratio)
                #sum_deformable_mask = sum_deformable_mask.detach()
                # logger.info(f"teacher_deformable_mask is: {s_deformable_mask.size()}")
                # logger.info(f"student_deformable_mask is: {torch.count_nonzero(s_deformable_mask)}")

                kd_feat_loss += dist2(t_feats[_i], self.adaptation_layers[_i](x[_i]), attention_mask=sum_attention_mask,
                                      channel_attention_mask=c_sum_attention_mask, fg_mask = Mask_fg) * 7e-5*6 #7e-5 * 6
                #kd_feat_loss += dist2(t_feats[_i], self.adaptation_layers[_i](x[_i]), deformable_adaptation=sum_deformable_mask) * 7e-5*6 #7e-5 * 6
                kd_channel_loss += torch.dist(torch.mean(t_feats[_i], [2, 3]),
                                              self.channel_wise_adaptation[_i](torch.mean(x[_i], [2, 3]))) * 4e-3*6 #4e-3 * 6
                t_spatial_pool = torch.mean(t_feats[_i], [1]).view(t_feats[_i].size(0), 1, t_feats[_i].size(2),
                                                                   t_feats[_i].size(3))
                s_spatial_pool = torch.mean(x[_i], [1]).view(x[_i].size(0), 1, x[_i].size(2),
                                                             x[_i].size(3))
                kd_spatial_loss += torch.dist(t_spatial_pool,
                                              self.spatial_wise_adaptation[_i](s_spatial_pool)) * 4e-3*6 #4e-3 * 6

        losses.update({'kd_feat_loss': kd_feat_loss})
        losses.update({'kd_channel_loss': kd_channel_loss})
        losses.update({'kd_spatial_loss': kd_spatial_loss})

        kd_nonlocal_loss = 0
        with torch.no_grad():
            t_feats = self.teacher_model.extract_feat(img)

        if t_feats is not None:
            for _i in range(len(t_feats)):
                s_relation = self.student_non_local[_i](x[_i])
                t_relation = self.teacher_non_local[_i](t_feats[_i])
                kd_nonlocal_loss += torch.dist(self.non_local_adaptation[_i](s_relation), t_relation, p=2)
        losses.update(kd_nonlocal_loss=kd_nonlocal_loss * 7e-5*6) ##7e-5 * 6

        return losses

    def cuda(self, device=None):
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to cuda when calling cuda function."""
        self.teacher_model.cuda(device=device)
        return super().cuda(device=device)

    def train(self, mode=True):
        """Set the same train mode for teacher and student model."""
        if self.eval_teacher:
            self.teacher_model.train(False)
        else:
            self.teacher_model.train(mode)
        super().train(mode)

    def __setattr__(self, name, value):
        """Set attribute, i.e. self.name = value
        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher_model':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)
