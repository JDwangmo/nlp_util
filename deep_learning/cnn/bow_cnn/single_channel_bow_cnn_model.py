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
from deep_learning.cnn.common import CnnBaseClass
from data_processing_util.feature_encoder.bow_feature_encoder import FeatureEncoder

class SingleChannelBowCNN(CnnBaseClass):
    """
        ## 简介：
        CNN(multi-channel BOW)模型，以 BOW 计数向量或 tfidf向量作为输入，以CNN为分类模型。
        该模型为双通道BOW的CNN模型，通道1为字的BOW向量通道，通道2为词（经过分词）的 BOW
        模型架构为：

            1. 输入层： shape 为： (1, vocabulary_size ,1)
            2. 多size的卷积和pooling层：
            3.

    """

    def __init__(self,
                 rand_seed=1337,
                 verbose=0,
                 feature_encoder=None,
                 optimizers='sgd',
                 input_length = None,
                 num_labels=None,
                 nb_epoch=100,
                 earlyStoping_patience=50,
                 l2_conv_filter_type =None,
                 l1_conv_filter_type = None,
                 full_connected_layer_units = None,
                 output_dropout_rate = 0.,
                 **kwargs
                 ):


        super(SingleChannelBowCNN, self).__init__(
            rand_seed=rand_seed,
            verbose=verbose,
            feature_encoder=feature_encoder,
            optimizers=optimizers,
            input_length=input_length,
            num_labels=num_labels,
            nb_epoch=nb_epoch,
            earlyStoping_patience=earlyStoping_patience,
            **kwargs
        )

        self.l1_conv_filter_type = l1_conv_filter_type
        self.l2_conv_filter_type = l2_conv_filter_type
        self.full_connected_layer_units = full_connected_layer_units
        self.output_dropout_rate = output_dropout_rate

        self.build_model()

    def create_network(self):
        '''
            创建单通道的 BOW-CNN（L）神经网络
                1. 输入层：( self.input_length, )
                2. reshape层：
                3. 第一层卷积层：多核卷积层:
                4. 单核卷积层
                5. flatten层
                6. 全连接层
                7. 输出Dropout层
                8. softmax分类层

        :return:
        '''


        from keras.layers import Input, Activation, merge, Flatten, Dropout,Reshape
        from keras.models import Model
        # from keras import backend as K

        # 1. 输入层：(1,self.input_length,1)
        l1_input_shape = ( self.input_length, )

        l1_input = Input(shape=l1_input_shape)

        # 2. reshape层
        l2_reshape = Reshape((1,l1_input_shape[0],1))(l1_input)
        # 3. 多核卷积层
        l3_conv = self.create_convolution_layer(
            input_layer=l2_reshape,
            convolution_filter_type=self.l1_conv_filter_type,
            )

        # model = Model(input=l1_input, output=[l2_conv_word])
        # model.summary()
        # quit()
        # 4. 单核卷积层
        l4_conv = self.create_convolution_layer(
            input_layer=l3_conv,
            convolution_filter_type=self.l2_conv_filter_type,
            )
        # 5. flatten层
        l5_flatten = Flatten()(l4_conv)
        # 6. 全连接层
        l6_full_connected_layer = self.create_full_connected_layer(
            input_layer=l5_flatten,
            units=self.full_connected_layer_units + [self.num_labels]
        )
        # 7. 输出Dropout层
        l6_dropout = Dropout(p=self.output_dropout_rate)(l6_full_connected_layer)

        # 8. softmax分类层
        l8_softmax_output = Activation("softmax")(l6_dropout)

        model = Model(input=l1_input, output=[l8_softmax_output])

        if self.verbose>0:
            model.summary()

        return model


    def print_model_descibe(self):
        import pprint
        detail = {'rand_seed': self.rand_seed,
                  'verbose': self.verbose,
                  'optimizers': self.optimizers,
                  'input_length': self.input_length,
                  'num_labels': self.num_labels,
                  'nb_epoch': self.nb_epoch,
                  'earlyStoping_patience': self.earlyStoping_patience,
                  'lr': self.lr,
                  'batch_size': self.batch_size,
                  'l2_conv_filter_type':self.l2_conv_filter_type,
                  'l1_conv_filter_type':self.l1_conv_filter_type,
                  'full_connected_layer_units':self.full_connected_layer_units,
                  'output_dropout_rate': self.output_dropout_rate,
                  }
        pprint.pprint(detail)
        logging.debug(detail)
        return detail


if __name__ == '__main__':
    train_X = ['你好', '无聊', '测试句子', '今天天气不错', '我要买手机']
    trian_y = [1, 3, 2, 2, 3]
    test_X = ['句子,句子', '你好', '你妹']
    test_y = [2, 3, 0]
    # 生成字词组合级别的特征
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
        feature_type='word_seg',
        max_features=2000,
    )
    train_X_feature = feature_encoder.fit_transform(train_X)
    test_X_feature = feature_encoder.transform(test_X)
    print(feature_encoder.vocabulary_size)
    print(','.join(feature_encoder.vocabulary))
    print(train_X_feature)
    print(test_X_feature)


    bow_cnn = SingleChannelBowCNN(
        rand_seed=1337,
        verbose=1,
        feature_encoder=feature_encoder,
        num_labels=4,
        input_length=feature_encoder.vocabulary_size,
        l1_conv_filter_type=[[5, 2, 1, 'valid',(-2,1),0.5],
                             [5, 4, 1, 'valid',(-2,1),0.],
                             [5, 6, 1, 'valid',(-2,1),0.],
                             ],
        l2_conv_filter_type = [[3, 3, 1, 'valid',(-2,1),0.]],
        full_connected_layer_units = [50,100],
        output_dropout_rate=0.5,
        nb_epoch=30,
        earlyStoping_patience=50,
        optimizers='sgd',
        batch_size=2,
    )
    bow_cnn.print_model_descibe()
    print(bow_cnn.fit(
        (train_X_feature, trian_y),
        (test_X_feature, test_y)))


    print(bow_cnn.predict('你好',transform_input=True))
    bow_cnn.accuracy((test_X_feature, test_y))
    print(bow_cnn.batch_predict(test_X,True))


    print(bow_cnn.batch_predict(test_X_feature, False))




