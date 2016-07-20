#encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-07-21'
    Email:   '383287471@qq.com'
    Describe: 
"""
from __future__ import print_function
from keras.engine.topology import Layer
from keras.layers import Convolution2D,MaxPooling2D
from keras import backend as K
import theano.tensor as T

class BowLayer(Layer):
    '''
        ## region onehot to bow layers
        # 将局部region的onehot转为bow编码，主要起到降低的目的，比如：
        句子： ‘有 什么 手机’
        onehot编码： [0001,1000,0100]
        步长为2时，输入bow向量为：[1001,1100]

    '''
    def __init__(self, size, **kwargs):
        self.size = size
        super(BowLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_batch = input_shape[0]
        self.num_channel = input_shape[1]
        self.input_length = input_shape[2]
        self.input_dim = input_shape[3]
        self.output_length = self.input_length - self.size + 1



    def call(self, x, mask=None):
        start = range(0, self.output_length)
        # print(start)
        y = []
        for s in start:
            # initial_weight_value =
            y.append( K.sum(x[:, :, s:s + 2, :],axis=2))
        y = K.concatenate(y,axis=2)
        return y

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1],self.output_length*input_shape[3])


class Convolution2DWrapper(Convolution2D):
    '''
        Convolution2D's wrapper.在原有的Convolution2D的基础上进行封装。使得可以创建更多样的卷积层。
        不同之处：
            1. 原本卷积核的行列只支持大于0的数值，即 nb_row > 0 and nb_col > 0.
            现进行修改，当nb_row为 -1 ，则取image的行大小为卷积核的行大小;当nb_col为 -1 ，则取image的列大小为卷积核的列大小。

    '''


    def build(self, input_shape):
        if self.nb_row==-1:
            # 假如nb_row为 -1 ，则取image的行大小为卷积核的行大小
            self.nb_row = input_shape[-2]

        if self.nb_col==-1:
            # 假如nb_col为 -1 ，则取image的列大小为卷积核的列大小
            self.nb_col = input_shape[-1]

        return super(Convolution2DWrapper, self).build(input_shape)

class MaxPooling2DWrapper(MaxPooling2D):
    '''
        - self.pool_size[0]>0的话，使用普通 max pooling，pool_size = k
        - self.pool_size[0]<0,使用 k-max pooling

    '''
    def build(self, input_shape):
        if self.pool_size[0] < 0:
            # k-max pooling
            self.k = abs(self.pool_size[0])
        elif self.pool_size[0] > 0:
            pass
        else:
            pass

    def _pooling_function(self, inputs, pool_size, strides, border_mode, dim_ordering):

        if pool_size[0]<0:
            # k-max pooling
            input_layer = T.transpose(inputs, axes=(0, 1, 3, 2))
            sorted_values = T.argsort(input_layer, axis=3)
            topmax_indexes = sorted_values[:, :, :, -self.k:]
            # sort indexes so that we keep the correct order within the sentence
            topmax_indexes_sorted = T.sort(topmax_indexes)

            # given that topmax only gives the index of the third dimension, we need to generate the other 3 dimensions
            dim0 = T.arange(0, input_layer.shape[0]).repeat(input_layer.shape[1] * input_layer.shape[2] * self.k)
            dim1 = T.arange(0, input_layer.shape[1]).repeat(self.k * input_layer.shape[2]).reshape((1, -1)).repeat(
                input_layer.shape[0],
                axis=0).flatten()
            dim2 = T.arange(0, input_layer.shape[2]).repeat(self.k).reshape((1, -1)).repeat(
                input_layer.shape[0] * input_layer.shape[1],
                axis=0).flatten()
            dim3 = topmax_indexes_sorted.flatten()
            x = T.transpose(
                input_layer[dim0, dim1, dim2, dim3].reshape(
                    (input_layer.shape[0], input_layer.shape[1], input_layer.shape[2], self.k)),
                axes=(0, 1, 3, 2))
            return x
        else:
            return super(MaxPooling2DWrapper, self)._pooling_function(inputs, pool_size, strides, border_mode, dim_ordering)

    def get_output_shape_for(self, input_shape):
        if self.pool_size[0]<0:
            return (input_shape[0], input_shape[1], self.k, input_shape[3])
        else:
            return super(MaxPooling2DWrapper, self).get_output_shape_for(input_shape)




