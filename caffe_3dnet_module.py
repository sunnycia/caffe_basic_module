import os.path as osp
from caffe_basic_module import *

## import 3d caffe
CAFFE_ROOT = '/data/sunnycia/saliency_on_videoset/Train/C3D-v1.1-tmp'
if osp.join(CAFFE_ROOT, 'python') not in sys.path:
    sys.path.insert(0, osp.join(CAFFE_ROOT, 'python'))
import caffe

def Conv3d(name, bottom, num_output, kernel_size, kernel_depth, stride,temporal_stride, pad,temporal_pad, lr_mult=1, weight_filler='msra', have_bias=False):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Convolution3D'
    layer.bottom.extend([bottom])
    layer.top.extend([name])
    layer.convolution3d_param.num_output = num_output
    layer.convolution3d_param.kernel_size.extend([kernel_size])
    layer.convolution3d_param.kernel_depth.extend([kernel_depth])
    layer.convolution3d_param.stride.extend([stride])
    layer.convolution3d_param.temporal_stride.extend([temporal_stride])
    layer.convolution3d_param.pad.extend([pad])
    layer.convolution3d_param.temporal_pad.extend([temporal_pad])
    layer.convolution3d_param.weight_filler.type = weight_filler
    layer.convolution3d_param.bias_term = have_bias
    layer.param.extend(_get_param(1, lr_mult))
    return layer

def Res3dBlock(name, bottom, dim, stride, temporal_stride, block_type=None):
    layers = []
    if block_type == 'no_preact':
        res_bottom = bottom
        # 1x1 conv at shortcut branch
        layers.append(Conv3d(name + '_branch1', res_bottom, dim*4, kernel_size=1, kernel_depth=1, stride=stride, temporal_stride=temporal_stride, pad=0, temporal_pad=0))
        layers.extend(Bn_Sc(name + '_branch1', layers[-1].top[0]))

        shortcut_top = layers[-1].top[0]
    elif block_type == 'both_preact':
        # layers.extend(Act(name + '_pre', bottom))
        # res_bottom = layers[-1].top[0]
        res_bottom=bottom
        # 1x1 conv at shortcut branch
        layers.append(Conv3d(name + '_branch1', res_bottom, dim*4, kernel_size=1, kernel_depth=1, stride=stride, temporal_stride=temporal_stride, pad=0, temporal_pad=0))
        layers.extend(Bn_Sc(name + '_branch1', layers[-1].top[0]))
        shortcut_top = layers[-1].top[0]
    else:
        shortcut_top = bottom
        res_bottom=bottom
    # residual branch: conv1 -> conv1_act -> conv2 -> conv2_act -> conv3
    layers.append(Conv3d(name + '_branch2a', res_bottom, dim, kernel_size=1, kernel_depth=1, stride=1, temporal_stride=1, pad=0, temporal_pad=0))
    layers.extend(Bn_Sc(name + '_branch2a', layers[-1].top[0]))
    layers.extend(Act(name + '_branch2a', layers[-1].top[0]))
    layers.append(Conv3d(name + '_branch2b', layers[-1].top[0], dim, kernel_size=3,kernel_depth=3,stride=stride,temporal_stride=temporal_stride, pad=1, temporal_pad=1))
    layers.extend(Bn_Sc(name + '_branch2b', layers[-1].top[0]))
    layers.extend(Act(name + '_branch2b', layers[-1].top[0]))
    layers.append(Conv3d(name + '_branch2c', layers[-1].top[0], dim*4, kernel_size=1, kernel_depth=1, stride=1, temporal_stride=1, pad=0, temporal_pad=0))
    layers.extend(Bn_Sc(name + '_branch2c', layers[-1].top[0]))
    # elementwise addition
    layers.append(Add(name, [shortcut_top, layers[-1].top[0]]))
    layers.extend(Act(name, layers[-1].top[0]))
    return layers

def Res3dLayer(name, bottom, num_blocks, dim, stride, temporal_stride, layer_type=None):
    assert num_blocks >= 1
    _get_name = lambda i: '{}{}'.format(name,chr(i+96))
    layers = []
    first_block_type = 'no_preact' if layer_type == 'first' else 'both_preact'
    layers.extend(Res3dBlock(_get_name(1), bottom, dim, stride=stride, temporal_stride=temporal_stride, first_block_type))
    for i in xrange(2, num_blocks+1):
        layers.extend(Res3dBlock(_get_name(i), layers[-1].top[0], dim, stride=1,temporal_stride=1))
    return layers