#encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-07-14'
    Email:   '383287471@qq.com'
    Describe: 
"""
from __future__ import print_function

import numpy as np
import theano.tensor as T
from base.common_model_class import CommonModel


class BowCNN(CommonModel):

    def __init__(self,
                 rand_seed=1337,
                 verbose=0,
                 input_length=None,
                 num_labels=None,
                 conv_filter_type=None,
                 k=1,
                 nb_epoch=100,
                 earlyStoping_patience=50,
                 ):
        '''

            初始化参数

        :param rand_seed:
        :param verbose:
        :param input_length:
        :param num_labels:
        :param conv_filter_type:
        :param k:
        :param nb_epoch:
        :param earlyStoping_patience:
        '''

        super(BowCNN, self).__init__(rand_seed, verbose)

        if rand_seed is not None:
            np.random.seed(1337)

    def model_from_pickle(self, path):
        super(BowCNN, self).model_from_pickle(path)

    def batch_predict_bestn(self, sentence, bestn=1):
        super(BowCNN, self).batch_predict_bestn(sentence, bestn)


    def print_model_descibe(self):
        super(BowCNN, self).print_model_descibe()

    def fit(self, train_data=None, validation_data=None):
        super(BowCNN, self).fit(train_data, validation_data)

    def accuracy(self, test_data):
        super(BowCNN, self).accuracy(test_data)

    def kmaxpooling(self, k=1):
        '''
            分别定义 kmax 的output 和output shape
            !但是k-max的实现用到Lambda,而pickle无法dump function对象,所以使用该模型的时候,保存不了模型,待解决.
        :param k: 设置 k-max 层 的k
        :type k: int
        :return:  Lambda
        '''

        def kmaxpooling_output(input):
            '''
                实现 k-max pooling
                    1. 先排序
                    2. 再分别取出前k个值
            :param k: k top higiest value
            :type k: int
            :return:
            '''
            input = T.transpose(input, axes=(0, 1, 3, 2))
            sorted_values = T.argsort(input, axis=3)
            topmax_indexes = sorted_values[:, :, :, -k:]
            # sort indexes so that we keep the correct order within the sentence
            topmax_indexes_sorted = T.sort(topmax_indexes)

            # given that topmax only gives the index of the third dimension, we need to generate the other 3 dimensions
            dim0 = T.arange(0, input.shape[0]).repeat(input.shape[1] * input.shape[2] * k)
            dim1 = T.arange(0, input.shape[1]).repeat(k * input.shape[2]).reshape((1, -1)).repeat(input.shape[0],
                                                                                                  axis=0).flatten()
            dim2 = T.arange(0, input.shape[2]).repeat(k).reshape((1, -1)).repeat(input.shape[0] * input.shape[1],
                                                                                 axis=0).flatten()
            dim3 = topmax_indexes_sorted.flatten()
            return T.transpose(
                input[dim0, dim1, dim2, dim3].reshape((input.shape[0], input.shape[1], input.shape[2], k)),
                axes=(0, 1, 3, 2))

        def kmaxpooling_output_shape(input_shape):
            return (input_shape[0], input_shape[1], k, input_shape[3])

        from keras.layers import Lambda
        return Lambda(kmaxpooling_output, kmaxpooling_output_shape, name='k-max')

    def create_multi_size_convolution_layer(self,
                                            input_shape=None,
                                            convolution_filter_type=None,
                                            k=1,
                                            ):
        """
            创建一个多类型（size，大小）核卷积层模型，可以直接添加到 keras的模型中去。
                1. 为每种size的核分别创建 Sequential 模型，模型内 搭建一个 2D卷积层 和一个 k-max pooling层
                2. 将1步骤创建的卷积核的结果 进行 第1维的合并，变成并行的卷积核
                3. 返回一个 4D 的向量

        必须是一个4D的输入，(n_batch,channel,row,col)

        :param convolution_filter_type: 卷积层的类型.一种 size对应一个 list

            for example:每个列表代表一种类型(size)的卷积核,
                conv_filter_type = [[100,2,word_embedding_dim,'valid'],
                                    [100,4,word_embedding_dim,'valid'],
                                    [100,6,word_embedding_dim,'valid'],
                                   ]
        :type convolution_filter_type: array-like
        :param input_shape: 输入的 shape，3D，类似一张图，(channel,row,col)比如 （1,5,5）表示单通道5*5的图片
        :type input_shape: array-like
        :param k: 设置 k-max 层 的 k
        :type k: int
        :return: convolution model，4D-array
        :rtype: Sequential
        """

        assert len(
            input_shape) == 3, 'warning: 因为必须是一个4D的输入，(n_batch,channel,row,col)，所以input shape必须是一个3D-array，(channel,row,col)!'

        from keras.layers import Convolution2D, Activation, MaxPooling2D, Merge
        from keras.models import Sequential
        # 构建第一层卷积层和1-max pooling
        conv_layers = []
        for items in convolution_filter_type:

            nb_filter, nb_row, nb_col, border_mode = items

            m = Sequential()
            m.add(Convolution2D(nb_filter,
                                nb_row,
                                nb_col,
                                border_mode=border_mode,
                                input_shape=input_shape,
                                ))
            m.add(Activation('relu'))

            # 1-max
            if k == 1:
                if border_mode == 'valid':
                    pool_size = (input_shape[1] - nb_row + 1, 1)
                elif border_mode == 'same':
                    pool_size = (input_shape[1], 1)
                else:
                    pool_size = (input_shape[1] - nb_row + 1, 1)
                m.add(MaxPooling2D(pool_size=pool_size, name='1-max'))
            else:
                # k-max pooling
                # todo
                # 因为kmax需要用到Lambda,而pickle无法dump function对象,所以使用该模型的时候,保存不了模型,待解决.
                m.add(self.kmaxpooling(k=k))
            # m.summary()
            conv_layers.append(m)

        # 卷积的结果进行拼接
        cnn_model = Sequential()
        cnn_model.add(Merge(conv_layers, mode='concat', concat_axis=1))
        # print(cnn_model.get_output_shape_at(-1))
        return cnn_model


    def build_model(self):
        '''
            构建 CNN-BOW 模型

        :return:
        '''

        from keras.layers import Input, Convolution2D, Activation, MaxPooling2D, Merge, Dense, Flatten, Dropout
        from keras.models import Sequential, Model
        from keras.optimizers import SGD


        self.create_multi_size_convolution_layer(input_shape=)
        quit()


        main_input_shape = (1, train_X_feature_4dim.shape[2], train_X_feature_4dim.shape[3])
        input = Input(shape=main_input_shape)
        seq = Sequential()

        conv_layers = [
            [
                Convolution2D(16, 5, main_input_shape[-1], border_mode='valid', input_shape=main_input_shape),
                Activation('tanh'),
                MaxPooling2D(pool_size=(2, 1))
            ],
            [
                Convolution2D(16, 10, main_input_shape[-1], border_mode='valid', input_shape=main_input_shape),
                Activation('tanh'),
                MaxPooling2D(pool_size=(2, 1))
            ],
            [Convolution2D(16, 20, main_input_shape[-1], border_mode='valid', input_shape=main_input_shape),
             Activation('tanh'),
             MaxPooling2D(pool_size=(2, 1))
             ]
        ]
        conv = []
        for layers in conv_layers:
            m = Sequential()
            [m.add(layer) for layer in layers]
            conv.append(m)

        seq.add(Merge(conv, mode='concat', concat_axis=2))
        seq.add(Convolution2D(32, 5, 1, border_mode='valid'))
        seq.add(MaxPooling2D(pool_size=(2, 1)))
        seq.add(Flatten())
        print
        seq.summary()
        # quit()
        # normal_input = BatchNormalization(mode=1,axis=1,beta_init='zero',gamma_init='one')(input)
        normal_input = input
        conv_output = seq([normal_input, normal_input, normal_input])
        l1 = Dense(output_dim=1000, init="glorot_uniform", activation='relu')(conv_output)
        l1 = Dense(output_dim=5000, init="glorot_uniform", activation='relu')(l1)
        l2 = Dense(output_dim=200, init="glorot_uniform", activation='relu')(l1)
        l2 = Dropout(p=0.9)(l2)
        # merge_layers = merge([l2,input2],mode='concat')
        # l3 = Dense(output_dim=200, init="glorot_uniform",activation='relu')(merge_layers)
        l3 = Dense(output_dim=len(index_to_label),
                   init="glorot_uniform",
                   activation='relu',
                   W_regularizer='l2')(l2)
        # l4 = Dropout(p=0.5)(l3)
        softmax_output = Activation("softmax")(l3)

        # model1 = Model(input=[input,input2], output=[softmax_output])
        model = Model(input=[input], output=[softmax_output])
        print
        model.summary()
        sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)

    def save_model(self, path):
        super(BowCNN, self).save_model(path)

    def batch_predict(self, sentence):
        super(BowCNN, self).batch_predict(sentence)

    def predict(self, sentence):
        super(BowCNN, self).predict(sentence)


if __name__ == '__main__':
    np.random.seed(0)
    print(np.random.randint(0,100))
    quit()
    bow_cnn = BowCNN()

