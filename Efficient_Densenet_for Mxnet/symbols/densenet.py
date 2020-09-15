'''
References:

Gao Huang, Zhuang Liu, Kilian Q. Weinberger, Laurens van der Maaten. "Densely Connected Convolutional Networks"
'''
#import find_mxnet
#assert find_mxnet
import mxnet as mx
import memonger
import math

def BasicBlock(stage_num, layer_num, data, growth_rate, stride, name, bottle_neck=True, drop_out=0.0, bn_mom=0.9, workspace=64):
    """Return BaiscBlock Unit symbol for building DenseBlock
    Parameters
    ----------
    data : str
        Input data
    growth_rate : int
        Number of output channels
    stride : tupe
        Stride used in convolution
    drop_out : float
        Probability of an element to be zeroed. Default = 0.2
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """

    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        bn1   = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name+'_x1_bn')
        act1  = mx.sym.Activation(data=bn1, act_type='relu', name='{0}_x1'.format(name.replace('conv', 'relu')))
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(growth_rate*4), kernel=(1,1), stride=(1,1), pad=(0,0),
                                      no_bias=True, workspace=workspace, name=name + '_x1')
        if drop_out > 0:
            conv1 = mx.symbol.Dropout(data=conv1, p=drop_out, name=name + '_dp1')
       
        bn2   = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_x2_bn')
        act2  = mx.sym.Activation(data=bn2, act_type='relu', name='{0}_x2'.format(name.replace('conv', 'relu')))
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(growth_rate), kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_x2')
        if drop_out > 0:
            conv2 = mx.symbol.Dropout(data=conv2, p=drop_out, name=name + '_dp2')
        return conv2
    else:
        bn1   = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_x1_bn')
        act1  = mx.sym.Activation(data=bn1, act_type='relu', name='{0}_x1'.format(name.replace('conv', 'relu')))
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(growth_rate), kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_x2')
        if drop_out > 0:
            conv1 = mx.symbol.Dropout(data=conv1, p=drop_out, name=name + '_dp1')
        return conv1
		
def DenseBlock(stage_num, units_num, data, growth_rate, name, bottle_neck=True, drop_out=0.0, bn_mom=0.9, workspace=64):
    """Return DenseBlock Unit symbol for building DenseNet
    Parameters
    ----------
    units_num : int
        the number of BasicBlock in each DenseBlock
    data : str	
        Input data
    growth_rate : int
        Number of output channels
    drop_out : float
        Probability of an element to be zeroed. Default = 0.2
    workspace : int
        Workspace used in convolution operator
    """

    for i in range(units_num):
        Block = BasicBlock(stage_num, i, data, growth_rate=growth_rate, stride=(1,1), name=name + '_%d' % (i+1), 
                            bottle_neck=bottle_neck, drop_out=drop_out, 
                            bn_mom=bn_mom, workspace=workspace)
        data = mx.symbol.Concat(data, Block, name='{0}_{1}'.format(name.replace('conv', 'concat_'), i+1))
        data._set_attr(mirror_stage='True')

    return data
	
def TransitionBlock(num_stage, stages, data, num_filter, stride, name, drop_out=0.0, bn_mom=0.9, workspace=64):
    """Return TransitionBlock Unit symbol for building DenseNet
    Parameters
    ----------
    num_stage : int
        Number of stage
    data : str
        Input data
    num : int
        Number of output channels
    stride : tupe
        Stride used in convolution
    name : str
        Base name of the operators
    drop_out : float
        Probability of an element to be zeroed. Default = 0.2
    workspace : int
        Workspace used in convolution operator
    """
    bn1   = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn')
 
    act1  = mx.sym.Activation(data=bn1, act_type='relu', name='{0}'.format(name.replace('conv','relu')))
    
    if num_stage == stages - 1:
        return act1
    conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter,
                                kernel=(1,1), stride=stride, pad=(0,0), no_bias=True,
                                workspace=workspace, name=name)
    if drop_out > 0:
        conv1 = mx.symbol.Dropout(data=conv1, p=drop_out, name=name + '_dp1')
    
    return mx.symbol.Pooling(conv1, global_pool=False, kernel=(2,2), stride=(2,2), pool_type='avg', name='{0}'.format(name.replace('conv','pool').replace('_blk','')))
 


def get_symbol(
    num_class,
    num_stage,
    units,
    growth_rate,
    drop_out=0.,
    l2_reg=2e-5,
    init_channels=64,
    workspace=64,
    bottle_neck=True,
    reduction=0.5,
    bn_mom=0.9
):
    n_channels = init_channels

    data = mx.symbol.Variable(name='data')
    body = mx.symbol.Convolution(
        name="conv1",
        data=data,
        num_filter=init_channels,
        kernel=(7, 7),
        stride=(2, 2),
        pad=(3, 3),
        no_bias=True
    )
    body = mx.symbol.BatchNorm(body, eps = l2_reg, name = 'conv1_bn')
    body = mx.symbol.Activation(data = body, act_type='relu', name='relu1')
    body = mx.symbol.Pooling(body, name='pool1', global_pool=False, kernel=(3,3), stride=(2,2), pool_type ='max')
    body._set_attr(mirror_stage='True')       

    for i in range(num_stage):
        body = DenseBlock(i+2, units[i], body, growth_rate=growth_rate, name='conv%d' % (i + 2), bottle_neck=bottle_neck, drop_out=drop_out, bn_mom=bn_mom, workspace=workspace)
        n_channels += units[i]*growth_rate
        n_channels = int(math.floor(n_channels*reduction))
        body = TransitionBlock(i, num_stage, body, n_channels, stride=(1,1), name='conv%d_%d_blk' %(i + 2, units[i]), drop_out=drop_out, bn_mom=bn_mom, workspace=workspace)
 
    body = mx.symbol.Pooling(body, global_pool=True, kernel=(8, 8), pool_type='avg', name = 'pool5')
    flat = mx.symbol.Flatten(data=body)
    fc = mx.symbol.FullyConnected(data=flat, num_hidden=num_class, name='fc1')

    return mx.symbol.SoftmaxOutput(data=fc, name='softmax')
