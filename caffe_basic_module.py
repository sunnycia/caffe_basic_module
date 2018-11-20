import numpy as np
import os
import os.path as osp
import sys
import google.protobuf as pb
import google.protobuf.text_format
from argparse import ArgumentParser
import math
# CAFFE_ROOT = osp.join(osp.dirname(__file__), '..', 'caffe')
# if osp.join(CAFFE_ROOT, 'python') not in sys.path:
#     sys.path.insert(0, osp.join(CAFFE_ROOT, 'python'))
import caffe
print caffe.__file__, 'in basic module'
from caffe.proto import caffe_pb2


python_loss_list = ['L1LossLayer', 'KLLossLayer', 'sKLLossLayer', 'iKLLossLayer', 'GBDLossLayer']
cpp_loss_list = ['KLDivLoss', 'EuclideanLoss', 'L1Loss', 'SigmoidCrossEntropyLoss', 'SoftmaxWithLoss']


def _get_include(phase):
    inc = caffe_pb2.NetStateRule()
    if phase == 'train':
        inc.phase = caffe_pb2.TRAIN
    elif phase == 'test':
        inc.phase = caffe_pb2.TEST
    else:
        raise ValueError("Unknown phase {}".format(phase))
    return inc

def _get_param(num_param, lr_mult=1):
    if num_param == 1:
        # only weight
        param = caffe_pb2.ParamSpec()
        param.lr_mult = 1*lr_mult
        param.decay_mult = 1*lr_mult
        return [param]
    elif num_param == 2:
        # weight and bias
        param_w = caffe_pb2.ParamSpec()
        param_w.lr_mult = 1*lr_mult
        param_w.decay_mult = 1*lr_mult
        param_b = caffe_pb2.ParamSpec()
        param_b.lr_mult = 2*lr_mult
        param_b.decay_mult = 0
        return [param_w, param_b]
    else:
        raise ValueError("Unknown num_param {}".format(num_param))

def _get_transform_param(phase):
    param = caffe_pb2.TransformationParameter()
    param.crop_size = 224
    param.mean_value.extend([104, 117, 123])
    param.force_color = True
    if phase == 'train':
        param.mirror = True
    elif phase == 'test':
        param.mirror = False
    else:
        raise ValueError("Unknown phase {}".format(phase))
    return param

def Data(name, tops, source, batch_size, phase):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Data'
    layer.top.extend(tops)
    layer.data_param.source = source
    layer.data_param.batch_size = batch_size
    layer.data_param.backend = caffe_pb2.DataParameter.LMDB
    layer.include.extend([_get_include(phase)])
    layer.transform_param.CopyFrom(_get_transform_param(phase))
    return layer

def Data_python(name, tops, param_str='2,3,600,800', bottom=None, module="CustomData", layer_name="CustomData"):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Python'
    layer.top.extend(tops)
    if bottom:
        layer.bottom.extend([bottom])
    layer.python_param.module = module
    layer.python_param.layer = layer_name
    layer.python_param.param_str = param_str
    return layer

def Conv(name, bottom, num_output, kernel_size, stride, pad, lr_mult=1, weight_filler='msra', have_bias=False):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Convolution'
    layer.bottom.extend([bottom])
    layer.top.extend([name])
    layer.convolution_param.num_output = num_output
    layer.convolution_param.kernel_size.extend([kernel_size])
    layer.convolution_param.stride.extend([stride])
    layer.convolution_param.pad.extend([pad])
    layer.convolution_param.weight_filler.type = weight_filler
    layer.convolution_param.bias_term = have_bias
    layer.param.extend(_get_param(1, lr_mult))
    return layer

def Dropout(name,bottom,dropout_ratio):
    dropout_layer = caffe_pb2.LayerParameter()
    dropout_layer.type = 'Dropout'
    dropout_layer.bottom.extend([bottom])
    dropout_layer.top.extend([name])
    dropout_layer.name = name
    dropout_layer.dropout_param.dropout_ratio=dropout_ratio

    return dropout_layer

def Conv_multi_bottom(name, bottom, num_output, kernel_size, stride, pad):
    # bottom and top are list
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Convolution'
    layer.bottom.extend(bottom)

    name_list = [name+'_'+str(i) for i in range(len(bottom))]
    layer.top.extend(name_list)
    layer.convolution_param.num_output = num_output
    layer.convolution_param.kernel_size.extend([kernel_size])
    layer.convolution_param.stride.extend([stride])
    layer.convolution_param.pad.extend([pad])
    layer.convolution_param.weight_filler.type = 'msra'
    layer.convolution_param.bias_term = False
    layer.param.extend(_get_param(1))
    return layer

def Bilinear_upsample(name, bottom, num_output, factor, lr_mult=1, weight_filler='bilinear',dilation=1):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Deconvolution'
    layer.bottom.extend([bottom])
    layer.top.extend([name])

    kernel_size = int(2*factor-factor%2)
    stride=factor
    pad=int(math.ceil((factor-1)/2.))

    kernel_size = int(2*temporal_factor-temporal_factor%2)
    temporal_stride=temporal_factor
    temporal_pad=int(math.ceil((temporal_factor-1)/2.))




    layer.convolution_param.num_output = num_output
    # layer.convolution_param.group = num_output
    layer.convolution_param.kernel_size.extend([kernel_size])
    layer.convolution_param.kernel_depth.extend([kernel_depth])
    layer.convolution_param.stride.extend([stride])
    layer.convolution_param.temporal_stride.extend([temporal_stride])
    layer.convolution_param.pad.extend([pad])
    layer.convolution_param.temporal_pad.extend([temporal_pad])
    layer.convolution_param.dilation.extend([dilation])
    layer.convolution_param.weight_filler.type = weight_filler
    layer.convolution_param.bias_term = False
    layer.param.extend(_get_param(1, lr_mult=lr_mult))
    return layer

def Eltwise(name, bottom_list, operation=2):
    # prod 0, sum 1, max 2
    eltwise_layer = caffe_pb2.LayerParameter()
    eltwise_layer.type = 'Eltwise'
    eltwise_layer.bottom.extend(bottom_list)
    eltwise_layer.top.extend([name])
    eltwise_layer.eltwise_param.operation=operation
    eltwise_layer.name = name

    return [eltwise_layer]

def Concat(name, bottom_list):
    concat_layer = caffe_pb2.LayerParameter()
    concat_layer.type = 'Concat'
    concat_layer.bottom.extend(bottom_list)
    concat_layer.top.extend([name])
    concat_layer.name = name

    return [concat_layer]

def Slice(name, bottom, slice_points, axis=1):
    slice_layer = caffe_pb2.LayerParameter()
    slice_layer.type = 'Slice'
    slice_layer.bottom.extend([bottom])
    slice_layer.slice_param.axis=axis
    # for i in range(len(slice_points)):
    #     slice_layer.slice_param.slice_point=slice_points[i]
    slice_layer.slice_param.slice_point.extend(slice_points)

    top_list = [bottom+'_%s'%str(i) for i in range(1, 2+len(slice_points))]
    # top_list = []
    # for i in range(len(slice_points)+1):
    #     top_list.append(bottom+'_%s'%str(i))
    slice_layer.top.extend(top_list)
    slice_layer.name = name+'_slice'

    return [slice_layer]

def Bn_Sc(name, bottom, keep_name=False):
    top_name=name
    name=name.replace('res', '')
    # BN

    bn_layer = caffe_pb2.LayerParameter()
    if not keep_name:
        bn_layer.name = 'bn' + name
    bn_layer.type = 'BatchNorm'
    bn_layer.bottom.extend([bottom])
    bn_layer.top.extend([top_name])
    # Scale
    scale_layer = caffe_pb2.LayerParameter()
    if not keep_name:
        scale_layer.name = 'scale'+name
    scale_layer.type = 'Scale'
    scale_layer.bottom.extend([top_name])
    scale_layer.top.extend([top_name])
    scale_layer.scale_param.filler.value = 1
    scale_layer.scale_param.bias_term = True
    scale_layer.scale_param.bias_filler.value = 0
    return [bn_layer, scale_layer]

def Act(name, bottom, act_type='ReLU'):
    top_name = name
    # ReLU
    relu_layer = caffe_pb2.LayerParameter()
    relu_layer.name = name + '_activation'
    relu_layer.type = act_type
    relu_layer.bottom.extend([top_name])
    relu_layer.top.extend([top_name])
    return [relu_layer]

def Pool(name, bottom, pooling_method, kernel_size, stride, pad):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Pooling'
    layer.bottom.extend([bottom])
    layer.top.extend([name])
    if pooling_method == 'max':
        layer.pooling_param.pool = caffe_pb2.PoolingParameter.MAX
    elif pooling_method == 'ave':
        layer.pooling_param.pool = caffe_pb2.PoolingParameter.AVE
    else:
        raise ValueError("Unknown pooling method {}".format(pooling_method))
    layer.pooling_param.kernel_size = kernel_size
    layer.pooling_param.stride = stride
    layer.pooling_param.pad = pad
    return layer

def Linear(name, bottom, num_output):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'InnerProduct'
    layer.bottom.extend([bottom])
    layer.top.extend([name])
    layer.inner_product_param.num_output = num_output
    layer.inner_product_param.weight_filler.type = 'msra'
    layer.inner_product_param.bias_filler.value = 0
    layer.param.extend(_get_param(2))
    return layer

def Add(name, bottoms):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Eltwise'
    layer.bottom.extend(bottoms)
    layer.top.extend([name])
    return layer

def ResBlock(name, bottom, dim, stride, block_type=None):
    layers = []
    if block_type == 'no_preact':
        res_bottom = bottom
        # 1x1 conv at shortcut branch
        layers.append(Conv(name + '_branch1', res_bottom, dim*4, 1, stride, 0))
        layers.extend(Bn_Sc(name + '_branch1', layers[-1].top[0]))

        shortcut_top = layers[-1].top[0]
    elif block_type == 'both_preact':
        # layers.extend(Act(name + '_pre', bottom))
        # res_bottom = layers[-1].top[0]
        res_bottom=bottom
        # 1x1 conv at shortcut branch
        layers.append(Conv(name + '_branch1', res_bottom, dim*4, 1, stride, 0))
        layers.extend(Bn_Sc(name + '_branch1', layers[-1].top[0]))
        shortcut_top = layers[-1].top[0]
    else:
        shortcut_top = bottom
        # preact at residual branch
        # layers.extend(Act(name + '_pre', bottom))
        # res_bottom = layers[-1].top[0]
        res_bottom=bottom
    # residual branch: conv1 -> conv1_act -> conv2 -> conv2_act -> conv3
    layers.append(Conv(name + '_branch2a', res_bottom, dim, 1, 1, 0))
    layers.extend(Bn_Sc(name + '_branch2a', layers[-1].top[0]))
    layers.extend(Act(name + '_branch2a', layers[-1].top[0]))
    layers.append(Conv(name + '_branch2b', layers[-1].top[0], dim, 3, stride, 1))
    layers.extend(Bn_Sc(name + '_branch2b', layers[-1].top[0]))
    layers.extend(Act(name + '_branch2b', layers[-1].top[0]))
    layers.append(Conv(name + '_branch2c', layers[-1].top[0], dim*4, 1, 1, 0))
    layers.extend(Bn_Sc(name + '_branch2c', layers[-1].top[0]))
    # elementwise addition
    layers.append(Add(name, [shortcut_top, layers[-1].top[0]]))
    layers.extend(Act(name, layers[-1].top[0]))
    return layers

def ResLayer(name, bottom, num_blocks, dim, stride, layer_type=None):
    assert num_blocks >= 1
    _get_name = lambda i: '{}{}'.format(name,chr(i+96))
    layers = []
    first_block_type = 'no_preact' if layer_type == 'first' else 'both_preact'
    layers.extend(ResBlock(_get_name(1), bottom, dim, stride, first_block_type))
    for i in xrange(2, num_blocks+1):
        layers.extend(ResBlock(_get_name(i), layers[-1].top[0], dim, 1))
    return layers

def LossLayer(name, bottoms, loss_type):
    loss_list = []
    weight_list=[]
    if '+' in loss_type:
        loss_list=loss_type.split('-')[0].split('+')
        weight_list = loss_type.split('-')[1].split('+')
        weight_list=[int(i) for i in weight_list]
    else:
        loss_list.append(loss_type)
        weight_list.append(1)
    layer_list=[]
    for loss,weight in zip(loss_list,weight_list):
        if loss in cpp_loss_list:
            layer_list.append(Loss(name, bottoms, loss_type=loss, weight=weight))
        elif loss in python_loss_list:
            layer_list.append(Loss_python(name, bottoms, loss=loss, weight=weight))
        else:
            raise NotImplementedError
        name=name+'_'
    return layer_list


def Loss(name, bottoms,loss_type='EuclideanLoss', weight=1):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = loss_type
    layer.bottom.extend(bottoms)
    layer.loss_weight.extend([weight])
    layer.top.extend([name])
    return layer

def Loss_python(name, bottoms, module="CustomLossFunction", loss="L1LossLayer",weight=1):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Python'
    layer.bottom.extend(bottoms)
    layer.top.extend([name])
    layer.loss_weight.extend([weight])
    layer.python_param.module=module
    layer.python_param.layer=loss
    return layer


def Accuracy(name, bottoms, top_k):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Accuracy'
    layer.bottom.extend(bottoms)
    layer.top.extend([name])
    layer.accuracy_param.top_k = top_k
    layer.include.extend([_get_include('test')])
    return layer
