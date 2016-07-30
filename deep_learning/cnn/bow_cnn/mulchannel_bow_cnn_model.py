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

class MultiChannelBowCNN(CnnBaseClass):
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
                 seg_feature_encoder=None,
                 word_feature_encoder=None,
                 optimizers='sgd',
                 num_labels=None,
                 nb_epoch=100,
                 earlyStoping_patience=50,
                 seg_input_length = None,
                 word_input_length = None,
                 l2_conv_filter_type =None,
                 l1_conv_word_filter_type = None,
                 l1_conv_seg_filter_type = None,
                 full_connected_layer_units = None,
                 **kwargs
                 ):


        super(MultiChannelBowCNN, self).__init__(
            rand_seed,
            verbose,
            None,
            optimizers,
            0,
            num_labels,
            nb_epoch,
            earlyStoping_patience,
            **kwargs)

        self.word_feature_encoder = word_feature_encoder
        self.seg_feature_encoder = seg_feature_encoder
        self.word_input_length = word_input_length
        self.seg_input_length = seg_input_length
        self.l1_conv_word_filter_type = l1_conv_word_filter_type
        self.l1_conv_seg_filter_type = l1_conv_seg_filter_type
        self.l2_conv_filter_type = l2_conv_filter_type
        self.full_connected_layer_units = full_connected_layer_units

        self.build_model()

    def fit(self, train_data=None, validation_data=None):
        '''
            cnn model 的训练
                1. 对数据进行格式转换,比如 转换 y 的格式:转成onehot编码
                2. 模型训练

        :param train_data: 训练数据,格式为:(train_X, train_y),train_X中每个句子以字典索引的形式表示(使用data_processing_util.feature_encoder.onehot_feature_encoder编码器编码),train_y是一个整形列表.
        :type train_data: (array-like,array-like)
        :param validation_data: 验证数据,格式为:(validation_X, validation_y),validation_X中每个句子以字典索引的形式表示(使用data_processing_util.feature_encoder.onehot_feature_encoder编码器编码),validation_y是一个整形列表.
        :type validation_data: (array-like,array-like)
        :return: None
        '''
        # -------------- region start : 1. 对数据进行格式转换,比如 转换 y 的格式:转成onehot编码 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('2. 对数据进行格式转换,比如 转换 y 的格式:转成onehot编码')
            print('2. 对数据进行格式转换,比如 转换 y 的格式:转成onehot编码')
        # -------------- code start : 开始 -------------


        [train_X1, train_X2], train_y = train_data
        train_X1 = np.asarray(train_X1)

        train_X2 = np.asarray(train_X2)

        # print(train_X.shape)
        [validation_X1, validation_X2], validation_y = validation_data
        validation_X1 = np.asarray(validation_X1)

        validation_X2 = np.asarray(validation_X2)


        return super(MultiChannelBowCNN,self).fit(
            ([train_X1,train_X2],train_y),
            ([validation_X1, validation_X2], validation_y))


    def create_network(self):


        from keras.layers import Input, Activation, merge, Flatten, Dropout,Reshape
        from keras.models import Model
        from keras import backend as K

        # 1. 输入层：(1,self.input_length,1)
        l1_input_word_shape = ( self.word_input_length, )
        l1_input_seg_shape = ( self.seg_input_length, )

        l1_input_word = Input(shape=l1_input_word_shape)
        l1_input_seg = Input(shape=l1_input_seg_shape)

        # 2. reshape层
        l2_reshape_word = Reshape((1,l1_input_word_shape[0],1))(l1_input_word)
        l2_reshape_seg = Reshape((1,l1_input_seg_shape[0],1))(l1_input_seg)

        l2_conv_word = self.create_convolution_layer(l2_reshape_word,
                                                     self.l1_conv_word_filter_type
                                                     )

        l2_conv_seg = self.create_convolution_layer(l2_reshape_seg,
                                                 self.l1_conv_seg_filter_type
                                                 )
        l3_merge = merge((l2_conv_word,l2_conv_seg),mode='concat',concat_axis=2)
        # model = Model(input=[l1_input_word,l1_input_seg], output=[l3_merge])
        # model.summary()
        # quit()
        l3_conv = self.create_convolution_layer(l3_merge,
                                                               self.l2_conv_filter_type
                                                               )
        # 4. flatten层
        l4_flatten = Flatten()(l3_conv)
        l5_full_connected_layer = self.create_full_connected_layer(
            input_layer=l4_flatten,
            units=self.full_connected_layer_units + [[self.num_labels,0.]]
        )

        # 6. softmax分类层
        l6_softmax_output = Activation("softmax")(l5_full_connected_layer)

        model = Model(input=[l1_input_word,l1_input_seg], output=[l6_softmax_output])

        # softmax层的输出
        # self.model_output = K.function([l1_input_word,l1_input_seg, K.learning_phase()], [l8_softmax_output])
        if self.verbose>0:
            model.summary()

        return model

    def transform(self, data):
        '''
            批量转换数据转换数据

        :param train_data: array-like,2D
        :return: feature
        '''

        word_feature = self.word_feature_encoder.transform(data)
        seg_feature = self.seg_feature_encoder.transform(data)
        # print(word_feature)
        # print(seg_feature)
        return [word_feature,seg_feature]


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
                  'seg_input_length': self.seg_input_length,
                  'word_input_length': self.word_input_length,
                  'l2_conv_filter_type':self.l2_conv_filter_type,
                  'l1_conv_word_filter_type':self.l1_conv_word_filter_type,
                  'l1_conv_seg_filter_type':self.l1_conv_seg_filter_type,
                  'full_connected_layer_units':self.full_connected_layer_units,
                  }
        pprint.pprint(detail)
        logging.debug(detail)
        return detail


if __name__ == '__main__':
    train_X = ['你好', '无聊', '测试句子', '今天天气不错', '我要买手机']
    trian_y = [1, 3, 2, 2, 3]
    test_X = ['句子,句子', '你好', '你妹']
    test_y = [2, 3, 0]
    # 生成词级别的特征
    seg_feature_encoder = FeatureEncoder(
        verbose=0,
        need_segmented=True,
        full_mode=True,
        remove_stopword=True,
        replace_number=True,
        lowercase=True,
        zhs2zht=True,
        remove_url=True,
        feature_method='bow',
        feature_type='seg',
        max_features=2000,
    )
    train_seg_X_feature = seg_feature_encoder.fit_transform(train_X)
    test_seg_X_feature = seg_feature_encoder.transform(test_X)
    print(seg_feature_encoder.vocabulary_size)
    print(','.join(seg_feature_encoder.vocabulary))
    print(train_seg_X_feature)
    print(test_seg_X_feature)

    # 生成字级别的特征
    from data_processing_util.feature_encoder.bow_feature_encoder import FeatureEncoder
    word_feature_encoder = FeatureEncoder(
        verbose=0,
        need_segmented=True,
        full_mode=True,
        remove_stopword=True,
        replace_number=True,
        lowercase=True,
        zhs2zht=True,
        remove_url=True,
        feature_method='bow',
        feature_type='word',
        max_features=2000,
    )
    train_word_X_feature = word_feature_encoder.fit_transform(train_X)
    test_word_X_feature = word_feature_encoder.transform(test_X)
    print(word_feature_encoder.vocabulary_size)
    print(','.join(word_feature_encoder.vocabulary))
    print(train_word_X_feature)
    print(test_word_X_feature)
    bow_cnn = MultiChannelBowCNN(
        rand_seed=1337,
        verbose=1,
        seg_feature_encoder=seg_feature_encoder,
        word_feature_encoder=word_feature_encoder,
        num_labels=4,
        seg_input_length=seg_feature_encoder.vocabulary_size,
        word_input_length=word_feature_encoder.vocabulary_size,
        l1_conv_word_filter_type=[[5, 2, 1, 'valid',(-2,1),0.5],
                             [5, 4, 1, 'valid',(-2,1),0.],
                             [5, 6, 1, 'valid',(-2,1),0.],
                             ],
        l1_conv_seg_filter_type=[[5, 2, 1, 'valid',(-2,1),0.5],
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
        ([train_word_X_feature,train_seg_X_feature],trian_y),
        ([test_word_X_feature,test_seg_X_feature],test_y)))


    print(bow_cnn.predict('你好',transform_input=True))
    bow_cnn.accuracy(([test_word_X_feature,test_seg_X_feature],test_y))
    print(bow_cnn.batch_predict(test_X,True))


    print(bow_cnn.batch_predict([test_word_X_feature,test_seg_X_feature],False))




