#! /usr/bin/env python
# coding : utf-8

"""ECO network, implemented in Gluon"""

import os
import time
from mxnet.context import gpu
from mxnet.gluon import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.contrib.nn import HybridConcurrent
from mxnet.gluon.nn import HybridSequential
from mxnet import base


# basic implement
def _make_basic_conv(**kwargs):
    out = MyHybridSequential(prefix='')
    out.add(nn.Conv2D(use_bias=False, **kwargs))
    out.add(nn.BatchNorm(epsilon=0.0001))
    out.add(nn.Activation('relu'))
    return out


def _make_branch(use_pool, *conv_settings):
    out = MyHybridSequential(prefix='')
    if use_pool == 'avg':
        out.add(nn.AvgPool2D(pool_size=3, strides=1, padding=1))
    elif use_pool == 'max':
        out.add(nn.MaxPool2D(pool_size=3, strides=2))
    setting_names = ['channels', 'kernel_size', 'strides', 'padding']
    for setting in conv_settings:
        kwargs = {}
        for i, value in enumerate(setting):
            if value is not None:
                kwargs[setting_names[i]] = value
        out.add(_make_basic_conv(**kwargs))
    return out


def _make_AB(pool_features, prefix):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None,
                             (64, 1, None, None)))
        if 32 == pool_features:
            out.add(_make_branch(None,
                                 (64, 1, None, None),
                                 (64, 3, None, 1)))
        elif 64 == pool_features:
            out.add(_make_branch(None,
                                 (64, 1, None, None),
                                 (96, 3, None, 1)))
        out.add(_make_branch(None,
                             (64, 1, None, None),
                             (96, 3, None, 1),
                             (96, 3, None, 1)))
        out.add(_make_branch('avg',
                             (pool_features, 1, None, None)))
    return out


def _make_D():
    out = HybridConcurrent(axis=1, prefix='')
    with out.name_scope():
        out.add(_make_basic_conv(channels=352, kernel_size=1, strides=1))
        out.add(_make_branch(None,
                             (192, 1, None, None),
                             (320, 3, None, 1)))
        out.add(_make_branch(None,
                             (160, 1, None, None),
                             (224, 3, None, 1),
                             (224, 3, None, 1)))
        out.add(_make_branch('avg',
                             (128, 1, None, None)))
    return out


def _make_2D(prefix):
    # branch 1, 2D
    branch_1 = MyHybridSequential(prefix=prefix)
    with branch_1.name_scope():
        # branch_1_split1
        branch_1_split1 = HybridConcurrent(axis=1)
        # output : (16, 320, 14, 14)
        branch_1_split1.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        # output : (16, 160, 14, 14)
        branch_1_split1.add(_make_branch(None,
                                         (128, 1, None, None),
                                         (160, 3, 2, 1)))
        # output : (16, 96, 14, 14)
        branch_1_split1.add(_make_branch(None,
                                         (64, 1, None, None),
                                         (96, 3, None, 1),
                                         (96, 3, 2, 1)))
        # 注意：在上述HybridConcurrent中执行的concat操作，只能在第二个维度上进行，否则会出错，由于W,H相同，只能在C上合并, 因此axis为1
        branch_1.add(branch_1_split1)
        # branch_1_split2
        branch_1_split2 = HybridConcurrent(axis=1)
        branch_1_split2.add(_make_basic_conv(channels=224, kernel_size=1, strides=1))
        branch_1_split2.add(_make_branch(None,
                                         (64, 1, None, None),
                                         (96, 3, None, 1)))
        branch_1_split2.add(_make_branch(None,
                                         (96, 1, None, None),
                                         (128, 3, None, 1),
                                         (128, 3, None, 1)))
        branch_1_split2.add(_make_branch('avg',
                                         (128, 1, None, None)))
        branch_1.add(branch_1_split2)
        # branch_1_split3
        branch_1_split3 = HybridConcurrent(axis=1)
        branch_1_split3.add(_make_basic_conv(channels=192, kernel_size=1, strides=1))
        branch_1_split3.add(_make_branch(None,
                                         (96, 1, None, None),
                                         (128, 3, None, 1)))
        branch_1_split3.add(_make_branch(None,
                                         (96, 1, None, None),
                                         (128, 3, None, 1),
                                         (128, 3, None, 1)))
        branch_1_split3.add(_make_branch('avg',
                                         (128, 1, None, None)))
        branch_1.add(branch_1_split3)
        # branch_1_split4
        branch_1_split4 = HybridConcurrent(axis=1)
        branch_1_split4.add(_make_basic_conv(channels=160, kernel_size=1, strides=1))
        branch_1_split4.add(_make_branch(None,
                                         (128, 1, None, None),
                                         (160, 3, None, 1)))
        branch_1_split4.add(_make_branch(None,
                                         (128, 1, None, None),
                                         (160, 3, None, 1),
                                         (160, 3, None, 1)))
        branch_1_split4.add(_make_branch('avg',
                                         (128, 1, None, None)))
        branch_1.add(branch_1_split4)
        # branch_1_split5
        branch_1_split5 = HybridConcurrent(axis=1)
        branch_1_split5.add(_make_basic_conv(channels=96, kernel_size=1, strides=1))
        branch_1_split5.add(_make_branch(None,
                                         (128, 1, None, None),
                                         (192, 3, None, 1)))
        branch_1_split5.add(_make_branch(None,
                                         (160, 1, None, None),
                                         (192, 3, None, 1),
                                         (192, 3, None, 1)))
        branch_1_split5.add(_make_branch('avg',
                                         (128, 1, None, None)))
        branch_1.add(branch_1_split5)
        # branch_1_split6
        branch_1_split6 = HybridConcurrent(axis=1)
        branch_1_split6.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        branch_1_split6.add(_make_branch(None,
                                         (128, 1, None, None),
                                         (192, 3, 2, 1)))
        branch_1_split6.add(_make_branch(None,
                                         (192, 1, None, None),
                                         (256, 3, None, 1),
                                         (256, 3, 2, 1)))
        branch_1.add(branch_1_split6)
        # branch_1_split7
        branch_1.add(_make_D())
        # branch_1_split8
        branch_1.add(_make_D())
        branch_1.add(nn.AvgPool2D(pool_size=7, strides=1))
        branch_1.add(nn.Dropout(0.6))
        branch_1.add(MyReshape(shape=(-1, 1, 16, 1024)))
        branch_1.add(nn.AvgPool2D(pool_size=(16, 1), strides=1))
        branch_1.add(MyReshape(shape=(-1, 1024)))

    return branch_1


def _make_3D(prefix):
    # branch 2, 3D
    # 3D layer, first two layer are same as three convolution layer's first two layer
    branch_2 = MyHybridSequential(prefix=prefix)
    with branch_2.name_scope():
        branch_2.add(_make_branch(None,
                                  (64, 1, None, None),
                                  (96, 3, None, 1)))
        branch_2.add(MyReshape(shape=(-1, 16, 96, 28, 28)))
        branch_2.add(MyTranspose(axes=(0, 2, 1, 3, 4)))
        branch_2.add(nn.Conv3D(channels=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding=(1, 1, 1)))
        # Block1
        block_1 = BlockV1(128)
        branch_2.add(block_1)

        branch_2.add(nn.BatchNorm(epsilon=0.0001))
        branch_2.add(nn.Activation('relu'))
        # Block2
        block_2 = BlockV2(256)
        branch_2.add(block_2)

        # Block3
        block_3 = BlockV1(256)
        branch_2.add(block_3)

        branch_2.add(nn.BatchNorm(epsilon=0.0001))
        branch_2.add(nn.Activation('relu'))
        # Block4
        block_4 = BlockV2(512)
        branch_2.add(block_4)
        # Block5
        block_5 = BlockV1(512)
        branch_2.add(block_5)

        branch_2.add(nn.BatchNorm(epsilon=0.0001))
        branch_2.add(nn.Activation('relu'))
        branch_2.add(nn.AvgPool3D(pool_size=(4, 7, 7), strides=(1, 1, 1)))
        branch_2.add(MyReshape(shape=(-1, 512)))
        branch_2.add(nn.Dropout(0.5))

    return branch_2


def _make_C(prefix):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_2D("2D_"))
        out.add(_make_3D("3D_"))
    return out


class BlockV1(HybridBlock):
    """ Component of 3D branch. """
    def __init__(self, channels_num, **kwargs):
        super(BlockV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.BatchNorm(epsilon=0.0001),
                      nn.Activation('relu'),
                      nn.Conv3D(channels=channels_num, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding=(1, 1, 1)),
                      nn.BatchNorm(epsilon=0.0001), nn.Activation('relu'),
                      nn.Conv3D(channels=channels_num, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding=(1, 1, 1)))

    def hybrid_forward(self, F, x, *args, **kwargs):
        residual = x
        x = self.body(x)

        return residual + x


class BlockV2(HybridBlock):
    """ Component of 3D branch. """
    def __init__(self, channels_num, **kwargs):
        super(BlockV2, self).__init__(**kwargs)
        self.body1 = nn.HybridSequential(prefix='')
        self.body1.add(nn.Conv3D(channels=channels_num, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding=(1, 1, 1)),
                       nn.BatchNorm(epsilon=0.0001),
                       nn.Activation('relu'),
                       nn.Conv3D(channels=channels_num, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding=(1, 1, 1)))
        self.body2 = nn.HybridSequential(prefix='')
        self.body2.add(nn.Conv3D(channels=channels_num, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding=(1, 1, 1)))

    def hybrid_forward(self, F, x, *args, **kwargs):
        x1 = self.body1(x)
        x2 = self.body2(x)

        return x1 + x2


class MyReshape(HybridBlock):
    """ Reshape layer """
    def __init__(self, shape, **kwargs):
        super(MyReshape, self).__init__(**kwargs)
        self.shape = shape

    def _alias(self):
        return 'reshape'

    def hybrid_forward(self, F, x, *args, **kwargs):
        return F.reshape(x, self.shape, name='reshape')

    def __repr__(self):
        s = '{name}(shape={shape})'
        return s.format(name=self.__class__.__name__, shape=self.shape)


class MyTranspose(HybridBlock):
    """ Transpose layer"""
    def __init__(self, axes, **kwargs):
        super(MyTranspose, self).__init__(**kwargs)
        self.axes = axes

    def _alias(self):
        return 'transpose'

    def hybrid_forward(self, F, x, *args, **kwargs):
        return F.transpose(x, self.axes, name='transpose')

    def __repr__(self):
        s = '{name}(axes={axes})'
        return s.format(name=self.__class__.__name__, axes=self.axes)


# Test Network
class MyHybridSequential(HybridSequential):
    def __init__(self, prefix=None, params=None):
        super(MyHybridSequential, self).__init__(prefix=prefix, params=params)

    def add(self, *blocks):
        """Adds block on top of the stack."""
        for block in blocks:
            self.register_child(block)

    def hybrid_forward(self, F, x):
        for block in self._children.values():
            print('*****************************')
            print("layer_name: {}, 输入的x: {}".format(block._name, x.shape))
            x = block(x)
            print("layer_name: {}, 输出的x: {}".format(block._name, x.shape))
            print('*****************************')
        return x

    def __repr__(self):
        HybridSequential.__repr__(self)

    def __getitem__(self, key):
        HybridSequential.__getitem__(self, key)

    def __len__(self):
        return len(self._children)


# Net
class Eco(HybridBlock):
    r"""Eco Full network:

    Parameters
    ----------
    classes : int, default 1000
        Number of classification classes.
    """

    def __init__(self, classes=63, **kwargs):
        super(Eco, self).__init__(**kwargs)
        with self.name_scope():
            self.features = MyHybridSequential(prefix='')
            self.features.add(_make_basic_conv(channels=64, kernel_size=7, strides=2, padding=3))
            self.features.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            self.features.add(_make_basic_conv(channels=64, kernel_size=1, strides=1))
            self.features.add(_make_basic_conv(channels=192, kernel_size=3, strides=1, padding=1))
            self.features.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            self.features.add(_make_AB(32, 'A_'))
            self.features.add(_make_AB(64, 'B_'))
            self.features.add(_make_C('C_'))

            self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        x = self.output(x)
        return x


# Constructor
def eco_full(pretrained=False, ctx=gpu(), root=os.path.join(base.data_dir(), '/path/to/json'), **kwargs):
    r"""Build ECO_Full network

    Parameters
    ----------
    pretrained : bool, default False
    ctx : Context, default GPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    """
    net = Eco(**kwargs)
    if pretrained:
        from mxnet.gluon.model_zoo.model_store import get_model_file
        net.load_parameters(get_model_file('eco_full_kinetics', root=root), ctx=ctx)
    return net


if __name__ == '__main__':
    import mxnet as mx
    model = Eco()
    model.initialize(init=mx.init.Xavier(), ctx=mx.gpu(0))
    x = mx.nd.random.normal(shape=(16, 48, 224, 224))
    x = x.as_in_context(mx.gpu(0))
    start = time.time()
    output = model(x)
    print('耗时：{}'.format(time.time() - start))
    print(output)
    print(output.shape)