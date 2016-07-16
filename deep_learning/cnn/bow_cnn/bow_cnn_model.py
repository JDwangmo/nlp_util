#encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-07-14'
    Email:   '383287471@qq.com'
    Describe: 
"""
from __future__ import print_function

import logging
import numpy as np
from base.common_model_class import CommonModel
from data_processing_util.feature_encoder.bow_feature_encoder import FeatureEncoder

class BowCNN(CommonModel):
    """
        CNN(BOW)模型，以 BOW 计数向量或 tfidf向量作为输入，以CNN为分类模型。
        模型架构为：

            1. 输入层： shape 为： (1, vocabulary_size ,1)
            2. 多size的卷积和pooling层：
            3.

    """

    def __init__(self,
                 rand_seed=1337,
                 verbose=0,
                 feature_encoder = None,
                 num_labels=None,
                 l1_conv_filter_type=None,
                 l2_conv_filter_type=None,
                 k=1,
                 full_connected_layer_units = [50],
                 output_dropout_rate = 0,
                 nb_epoch=100,
                 earlyStoping_patience=50,
                 optimizers='sgd',
                 **kwargs
                 ):
        '''

            1. 初始化参数，并检验参数合法性。
            2. 设置随机种子，构建模型

        :param rand_seed: 随机种子,假如设置为为None时,则随机取随机种子
        :type rand_seed: int
        :param verbose: 数值越大,输出更详细的信息
        :type verbose: int
        :param feature_encoder: 输入数据的设置选项，设置输入编码器
        :type feature_encoder: bow_feature_encoder.FeatureEncoder
        :param num_labels: cnn设置选项,最后输出层的大小,即分类类别的个数.
        :type num_labels: int
        :param l1_conv_filter_type: cnn设置选项,第一层卷积层的类型：多size类型

            for example:每个列表代表一种类型(size)的卷积核,
                conv_filter_type = [[100,2,1,'valid'],
                                    [100,4,1,'valid'],
                                    [100,6,1,'valid'],
                                   ]

        :type l1_conv_filter_type: array-like
        :param l2_conv_filter_type: cnn设置选项,第二层卷积层的类型：单size类型。

            for example:每个列表代表一种类型(size)的卷积核,
                conv_filter_type = [[100,2,1,'valid']

        :type l2_conv_filter_type: array-like
        :param k: cnn设置选项,k-max pooling层的的k值,即设置要获取 前k个 值 ,默认为 1-max
        :type k: int
        :param output_dropout_rate: cnn设置选项,dropout层的的dropout rate,对输出层进入dropuout,如果为0,则不dropout
        :type output_dropout_rate: float
        :param nb_epoch: cnn设置选项,cnn迭代的次数.
        :type nb_epoch: int
        :param earlyStoping_patience: cnn设置选项,earlyStoping的设置,如果迭代次数超过这个耐心值,依旧不下降,则stop.
        :type earlyStoping_patience: int
        '''

        super(BowCNN, self).__init__(rand_seed, verbose)

        self.feature_encoder = feature_encoder
        self.optimizers = optimizers
        self.verbose = verbose
        self.num_labels = num_labels
        self.l1_conv_filter_type = l1_conv_filter_type
        self.l2_conv_filter_type = l2_conv_filter_type
        self.k = k
        self.full_connected_layer_units = full_connected_layer_units
        self.output_dropout_rate = output_dropout_rate
        self.nb_epoch = nb_epoch
        self.earlyStoping_patience = earlyStoping_patience
        self.kwargs = kwargs

        assert optimizers in ['sgd', 'adadelta'], 'optimizers只能取 sgd, adadelta！'

        assert feature_encoder is not None,'请先设置feature encoder（bow_feature_encoder.FeatureEncoder）！'

        # 输入的长度，即BOW向量的长度，为字典大小
        self.input_length = feature_encoder.vocabulary_size
        # cnn model
        self.model = None
        # 优化设置选项
        # 1. 模型学习率
        self.lr = self.kwargs.get('lr', 1e-2)
        # 批量大小
        self.batch_size = self.kwargs.get('batch_size', 32)
        if rand_seed is not None:
            np.random.seed(1337)
        self.build_model()

    def model_from_pickle(self, path):
        super(BowCNN, self).model_from_pickle(path)

    def batch_predict_bestn(self, sentence, bestn=1):
        super(BowCNN, self).batch_predict_bestn(sentence, bestn)


    def print_model_descibe(self):
        import pprint
        detail = {'rand_seed': self.rand_seed,
                  'verbose': self.verbose,
                  'optimizers': self.optimizers,
                  'input_length': self.input_length,
                  'num_labels': self.num_labels,
                  'l1_conv_filter_type': self.l1_conv_filter_type,
                  'l2_conv_filter_type': self.l2_conv_filter_type,
                  'kmaxpooling_k': self.k,
                  'output_dropout_rate': self.output_dropout_rate,
                  'nb_epoch': self.nb_epoch,
                  'earlyStoping_patience': self.earlyStoping_patience,
                  'lr': self.lr,
                  'batch_size': self.batch_size,
                  }
        pprint.pprint(detail)
        logging.debug(detail)
        return detail

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
            elif k == 0:
                m.add(MaxPooling2D(pool_size=(2, 1)))
            else:
                # k-max pooling
                # todo
                # 因为kmax需要用到Lambda,而pickle无法dump function对象,所以使用该模型的时候,保存不了模型,待解决.
                m.add(self.kmaxpooling(k=k))
            # m.summary()
            conv_layers.append(m)

        # 卷积的结果进行拼接
        cnn_model = Sequential()
        cnn_model.add(Merge(conv_layers, mode='concat', concat_axis=2))
        # cnn_model.summary()
        print(cnn_model.get_output_shape_at(-1))
        return cnn_model

    def creat_convolution_layer(
            self,
            input_shape = None,
            input=None,
            convolution_filter_type=None,
            k =None,
    ):
        '''
            创建一个卷积层模型，在keras的Convolution2D基础进行封装，使得可以创建多size和多size的卷积层

        :param input_shape: 上一层的shape
        :param input: 上一层
        :param convolution_filter_type: 卷积核类型，可以多size和单size，比如：
            1. 多size：每个列表代表一种类型(size)的卷积核,
                conv_filter_type = [[100,2,1,'valid',(k,1)],
                                    [100,4,1,'valid'],
                                    [100,6,1,'valid'],
                                   ]
            2. 单size：一个列表即可。[[100,2,1,'valid',(k,1)]]
        :param k: k-max-pooling 的 k值
        :return: kera TensorVariable,output,output_shape
        '''
        from keras.layers import Convolution2D,MaxPooling2D
        from keras.models import Sequential

        assert input_shape is not None,'input shape 不能为空！'
        if len(convolution_filter_type) == 1:
            nb_filter, nb_row, nb_col, border_mode,k = convolution_filter_type[0]
            # 单size 卷积层
            output_layer = Sequential()
            output_layer.add(Convolution2D(nb_filter, nb_row, nb_col, border_mode=border_mode,input_shape= input_shape))
            output_layer.add(MaxPooling2D(pool_size=k))
            output = output_layer(input)
            output_shape = output_layer.get_output_shape_at(-1)
        else:
            # 多size 卷积层
            output_layer = self.create_multi_size_convolution_layer(
                input_shape=input_shape,
                convolution_filter_type=convolution_filter_type,
                k=k,
            )
            output_shape = output_layer.get_output_shape_at(-1)
            output = output_layer([input] * len(convolution_filter_type))

        return output,output_shape[1:]

    def create_full_connected_layer(
            self,
            input= None,
            input_shape =None,
            units = None,
    ):
        '''
            创建多层的全连接层

        :param input_shape: 上一层的shape
        :param input: 上一层
        :param units: 每一层全连接层的单元数，比如:[100,20]
        :type units: array-like
        :return: output, output_shape
        '''

        from keras.models import Sequential
        from keras.layers import Dense

        output_layer = Sequential()
        output_layer.add(Dense(
            output_dim=units[0],
            init="glorot_uniform",
            activation='relu',
            input_shape = (input_shape,)
        ))

        for unit in units[1:]:
            output_layer.add(Dense(output_dim=unit, init="glorot_uniform", activation='relu'))

        output = output_layer(input)
        output_shape = output_layer.get_output_shape_at(-1)

        return output, output_shape[1:]


    def build_model(self):
        '''
            构建 CNN-BOW 模型

            1. 输入层：(1,self.input_length,1)
            2. 第一层卷积层：多size卷积层
            3. 第二层卷积层：单size卷积层
            4. max pooling 层
            5. flatten层
            6. hidden层：1000 units
            7. hidden层：5000 units
            8. hidden层：200 units

        :return:
        '''

        from keras.layers import Input, Activation, Merge, Dense, Flatten, Dropout,BatchNormalization
        from keras.models import Model
        from keras.optimizers import SGD
        from keras.callbacks import EarlyStopping

        # 1. 输入层：(1,self.input_length,1)
        l1_input_shape = (1,self.input_length,1)
        l1_input = Input(shape=l1_input_shape)

        # 2. 多size卷积层
        l2_conv,l2_conv_output_shape = self.creat_convolution_layer(
            input_shape=l1_input_shape,
            input=l1_input,
            convolution_filter_type=self.l1_conv_filter_type,
            k=self.k,
        )
        # 3. 单size卷积层 和 max pooling 层
        l3_conv,l3_conv_output_shape = self.creat_convolution_layer(
            input_shape=l2_conv_output_shape,
            input=l2_conv,
            convolution_filter_type=self.l2_conv_filter_type,
            k=self.k,
        )


        # 4. flatten层
        l4_flatten = Flatten()(l3_conv)
        print(l3_conv_output_shape)

        print(np.prod(l3_conv_output_shape))
        # 5. 全连接层
        l5_full_connected_layer,l5_full_connected_layer_output_shape = self.create_full_connected_layer(
            input = l4_flatten,
            input_shape= np.prod(l3_conv_output_shape),
            units=self.full_connected_layer_units,
        )
        quit()


        l6_dropout = Dropout(p=0.9)(l5_full_connected_layer)

        l7_output = Dense(output_dim=self.num_labels,
                   init="glorot_uniform",
                   activation='relu',
                   W_regularizer='l2')(l6_dropout)
        # l4 = Dropout(p=0.5)(l3)
        l8_softmax_output = Activation("softmax")(l7_output)

        # model1 = Model(input=[input,input2], output=[softmax_output])
        model = Model(input=[l1_input], output=[l12_softmax_output])
        model.summary()
        # -------------- region start : 3. 设置优化算法,earlystop等 -------------
        logging.debug('-' * 20)
        print('-' * 20)
        if self.verbose > 1:
            logging.debug('3. 设置优化算法,earlystop等')
            print('3. 设置优化算法,earlystop等')
        # -------------- code start : 开始 -------------


        if self.optimizers == 'sgd':
            optimizers = SGD(lr=self.lr, decay=1e-6, momentum=0.9, nesterov=True)
        elif self.optimizers == 'adadelta':
            optimizers = 'adadelta'
        else:
            optimizers = 'adadelta'
        model.compile(loss='categorical_crossentropy', optimizer=optimizers, metrics=['accuracy'])
        self.early_stop = EarlyStopping(patience=self.earlyStoping_patience, verbose=self.verbose)

        # -------------- code start : 结束 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
        # -------------- region end : 3. 设置优化算法,earlystop等 ---------------

        self.model = model

    def save_model(self, path):
        super(BowCNN, self).save_model(path)

    def batch_predict(self, sentence):
        super(BowCNN, self).batch_predict(sentence)

    def predict(self, sentence):
        super(BowCNN, self).predict(sentence)


if __name__ == '__main__':
    train_X = ['你好', '无聊', '测试句子', '今天天气不错', '我要买手机']
    trian_y = [1, 3, 2, 2, 3]
    test_X = ['句子', '你好', '你妹']
    test_y = [2, 3, 0]

    feature_encoder = FeatureEncoder(
        verbose=0,
        need_segmented=True,
        full_mode=True,
        remove_stopword=True,
        replace_number=True,
        lowercase=True,
        zhs2zht=True,
        remove_url=True,
        feature_method='bow',
        max_features=2000,
    )
    train_X_feature = feature_encoder.fit_transform(train_X)
    test_X_feature = feature_encoder.transform(test_X)
    print(','.join(feature_encoder.vocabulary))
    print(train_X_feature)
    bow_cnn = BowCNN(
        rand_seed=1337,
        verbose=0,
        feature_encoder=feature_encoder,
        num_labels=4,
        l1_conv_filter_type=[[10, 3, 1, 'valid'],
                             [10, 4, 1, 'valid'],
                             [10, 5, 1, 'valid'],
                             ],
        l2_conv_filter_type = [[10, 2, 1, 'valid',(3,1)]],
        k=0,
        full_connected_layer_units = [50,100],
        output_dropout_rate=0,
        nb_epoch=100,
        earlyStoping_patience=50,
        optimizers='sgd',
        batch_size=3,
    )
    bow_cnn.print_model_descibe()
    quit()

