import math
import os.path as osp
import sys
import caffe_basic_module as cbm
## import 3d caffe
# CAFFE_ROOT = '/data/sunnycia/saliency_on_videoset/Train/C3D-v1.1-tmp'
# if osp.join(CAFFE_ROOT, 'python') not in sys.path:
#     sys.path.insert(0, osp.join(CAFFE_ROOT, 'python'))
import caffe
print caffe.__file__, 'in 3d module'
from caffe.proto import caffe_pb2

def Conv3d(name, bottom, num_output, kernel_size, kernel_depth, stride,temporal_stride, pad,temporal_pad, lr_mult=1, weight_filler='msra', have_bias=False):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Convolution3D'
    layer.bottom.extend([bottom])
    layer.top.extend([name])
    layer.convolution3d_param.num_output = num_output
    layer.convolution3d_param.kernel_size = kernel_size
    layer.convolution3d_param.kernel_depth = kernel_depth
    layer.convolution3d_param.stride = stride
    layer.convolution3d_param.temporal_stride = temporal_stride
    layer.convolution3d_param.pad = pad
    layer.convolution3d_param.temporal_pad = temporal_pad
    layer.convolution3d_param.weight_filler.type = weight_filler
    layer.convolution3d_param.bias_term = have_bias
    layer.param.extend(cbm._get_param(1, lr_mult))
    return layer

def Bilinear_upsample_3d(name, bottom, num_output, factor, temporal_factor, lr_mult=1, weight_filler='bilinear'):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Deconvolution3D'
    layer.bottom.extend([bottom])
    layer.top.extend([name])

    kernel_size = int(2*factor-factor%2)
    stride=factor
    pad=int(math.ceil((factor-1)/2.))

    kernel_depth = int(2*temporal_factor-temporal_factor%2)
    temporal_stride=temporal_factor
    temporal_pad=int(math.ceil((temporal_factor-1)/2.))

    layer.convolution3d_param.num_output = num_output
    # layer.convolution3d_param.group = num_output
    layer.convolution3d_param.kernel_size = kernel_size
    layer.convolution3d_param.kernel_depth = kernel_depth
    layer.convolution3d_param.stride = stride
    layer.convolution3d_param.temporal_stride = temporal_stride
    layer.convolution3d_param.pad = pad
    layer.convolution3d_param.temporal_pad = temporal_pad
    # layer.convolution3d_param.dilation = dilation
    layer.convolution3d_param.weight_filler.type = weight_filler
    layer.convolution3d_param.bias_term = False
    layer.param.extend(cbm._get_param(1, lr_mult=lr_mult))
    return layer

def Res3dBlock(name, bottom, dim, stride, temporal_stride, block_type=None):
    layers = []
    if block_type == 'no_preact':
        res_bottom = bottom
        # 1x1 conv at shortcut branch
        layers.append(Conv3d(name + '_branch1', res_bottom, dim, kernel_size=1, kernel_depth=1, stride=stride, temporal_stride=temporal_stride, pad=0, temporal_pad=0))
        layers.extend(cbm.Bn_Sc(name + '_branch1', layers[-1].top[0]))

        shortcut_top = layers[-1].top[0]
    elif block_type == 'both_preact':
        # layers.extend(cbm.Act(name + '_pre', bottom))
        # res_bottom = layers[-1].top[0]
        res_bottom=bottom
        # 1x1 conv at shortcut branch
        layers.append(Conv3d(name + '_branch1', res_bottom, dim, kernel_size=1, kernel_depth=1, stride=stride, temporal_stride=temporal_stride, pad=0, temporal_pad=0))
        layers.extend(cbm.Bn_Sc(name + '_branch1', layers[-1].top[0]))
        shortcut_top = layers[-1].top[0]
    else:
        shortcut_top = bottom
        res_bottom=bottom
    # residual branch: conv1 -> conv1_act -> conv2 -> conv2_act -> conv3
    layers.append(Conv3d(name + '_branch2a', res_bottom, dim, kernel_size=3, kernel_depth=3, stride=stride, temporal_stride=temporal_stride, pad=1, temporal_pad=1))
    layers.extend(cbm.Bn_Sc(name + '_branch2a', layers[-1].top[0]))
    layers.extend(cbm.Act(name + '_branch2a', layers[-1].top[0]))
    layers.append(Conv3d(name + '_branch2b', layers[-1].top[0], dim, kernel_size=3,kernel_depth=3,stride=1,temporal_stride=1, pad=1, temporal_pad=1))
    layers.extend(cbm.Bn_Sc(name + '_branch2b', layers[-1].top[0]))
    layers.extend(cbm.Act(name + '_branch2b', layers[-1].top[0]))
    # layers.append(Conv3d(name + '_branch2c', layers[-1].top[0], dim, kernel_size=1, kernel_depth=1, stride=1, temporal_stride=1, pad=0, temporal_pad=0))
    # layers.extend(cbm.Bn_Sc(name + '_branch2c', layers[-1].top[0]))
    # elementwise addition
    layers.append(cbm.Add(name, [shortcut_top, layers[-1].top[0]]))
    layers.extend(cbm.Act(name, layers[-1].top[0]))
    return layers

def Res3dLayer(name, bottom, num_blocks, dim, stride, temporal_stride, layer_type=None):
    assert num_blocks >= 1
    _get_name = lambda i: '{}{}'.format(name,chr(i+96))
    layers = []
    first_block_type = 'no_preact' if layer_type == 'first' else 'both_preact'
    layers.extend(Res3dBlock(_get_name(1), bottom, dim, stride=stride, temporal_stride=temporal_stride, block_type=first_block_type))
    for i in xrange(2, num_blocks+1):
        layers.extend(Res3dBlock(_get_name(i), layers[-1].top[0], dim, stride=1,temporal_stride=1))
    return layers