# encoding=utf8

__author__ = 'jdwang'
__date__ = 'create date: 2016-06-23'
__email__ = '383287471@qq.com'

import numpy as np
from deep_learning.cnn.common import CnnBaseClass
import logging
import cPickle as pickle
from data_processing_util.feature_encoder.onehot_feature_encoder import FeatureEncoder
import theano.tensor as T
from sklearn.metrics import f1_score



class OnehotCNN(CnnBaseClass):
    '''
        一层CNN模型,随机初始化词向量,CNN-seq模型.借助Keras和jieba实现。
        架构各个层次分别为: 输入层,卷积层,1-max pooling层,全连接层,dropout层,softmax层
        具体见:
            https://github.com/JDwangmo/coprocessor#2convolutional-neural-networks-for-sentence-classification
        包含以下主要函数：
            1. build_model：构建模型
            2. fit：拟合和训练模型
            3. transform：对输入进行转换
            4. predict： 单句预测
            5. batch_predict： 批量预测
            6. save_model：保存模型
            7. model_from_pickle：恢复模型
            8. accuracy：模型验证
            9. print_model_descibe：打印模型详情
    '''

    def __init__(self,
                 rand_seed=1337,
                 verbose=0,
                 feature_encoder=None,
                 full_connected_layer_units=[50],
                 optimizers='sgd',
                 input_length=None,
                 num_labels=None,
                 conv1_filter_type=None,
                 conv2_filter_type=None,
                 output_dropout_rate=0.5,
                 nb_epoch=100,
                 earlyStoping_patience=50,
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
        :type feature_encoder: onehot_feature_encoder.FeatureEncoder
        :param optimizers: 求解的优化算法，目前支持: ['sgd','adadelta']
        :type optimizers: str
        :param input_length: cnn设置选项,输入句子(序列)的长度.
        :type input_length: int
        :param num_labels: cnn设置选项,最后输出层的大小,即分类类别的个数.
        :type num_labels: int
        :param conv1_filter_type: cnn设置选项,卷积层的类型.

            for example:每个列表代表一种类型(size)的卷积核,
                conv1_filter_type = [[100,2,word_embedding_dim,'valid',(1,1)],
                                    [100,4,word_embedding_dim,'valid',(1,1)],
                                    [100,6,word_embedding_dim,'valid',(1,1)],
                                   ]

        :type conv1_filter_type: array-like
        :param output_dropout_rate: cnn设置选项,dropout层的的dropout rate,对输出层进入dropuout,如果为0,则不dropout
        :type output_dropout_rate: float
        :param nb_epoch: cnn设置选项,cnn迭代的次数.
        :type nb_epoch: int
        :param earlyStoping_patience: cnn设置选项,earlyStoping的设置,如果迭代次数超过这个耐心值,依旧不下降,则stop.
        :type earlyStoping_patience: int
        '''
        CnnBaseClass.__init__(
            self,
            rand_seed=rand_seed,
            verbose=verbose,
            feature_encoder=feature_encoder,
            optimizers=optimizers,
            input_length=input_length,
            num_labels=num_labels,
            nb_epoch=nb_epoch,
            earlyStoping_patience=earlyStoping_patience,
        )
        self.input_length = input_length

        self.conv1_filter_type = conv1_filter_type
        self.conv2_filter_type = conv2_filter_type
        self.full_connected_layer_units = full_connected_layer_units
        self.output_dropout_rate = output_dropout_rate
        self.kwargs = kwargs

        self.conv1_feature_output = None

        # 构建模型
        self.build_model()

    def create_network(self):
        '''
            1. 创建 CNN 网络

                1. 输入层，2D，（n_batch,input_length,input_dim）
                2. Reshape层： 将embedding转换4-dim的shape，4D
                3. 第一层多size卷积层（含1-max pooling），使用三种size.
                4. Flatten层： 卷积的结果进行拼接,变成一列隐含层
                5. output hidden层
                6. output Dropout层
                7. softmax 分类层
            2. compile模型

        :return: cnn model network
        '''

        from keras.layers import Embedding, Input, Activation, Reshape, Dropout, Dense, Flatten
        from keras.models import Model
        from keras import backend as K

        # 1. 输入层
        l1_input_shape = ( self.input_length,self.feature_encoder.vocabulary_size)
        l1_input = Input(shape=l1_input_shape)

        # 2. Reshape层： 将embedding转换4-dim的shape
        l2_reshape= Reshape((1, self.input_length, self.feature_encoder.vocabulary_size))(l1_input)
        # 3. 第一层卷积层：多size卷积层（含1-max pooling），使用三种size.
        l3_cnn_model,l3_cnn_model_out_shape = self.create_convolution_layer(
            input_shape=(1,
                         self.input_length,
                         self.feature_encoder.vocabulary_size,
                         ),
            convolution_filter_type=self.conv1_filter_type,
            input=l2_reshape,
        )

        # 4. 第二层卷积层：单size卷积层 和 max pooling 层
        l4_conv, l4_conv_output_shape = self.create_convolution_layer(
            input_shape=l3_cnn_model_out_shape,
            input=l3_cnn_model,
            convolution_filter_type=self.conv2_filter_type,
        )

        # 5. Flatten层： 卷积的结果进行拼接,变成一列隐含层
        l5_flatten = Flatten()(l4_conv)
        # 6. 全连接层

        l6_full_connected_layer,l6_full_connected_layer_output_shape = self.create_full_connected_layer(
            input=l5_flatten,
            input_shape=np.prod(l4_conv_output_shape),
            units=self.full_connected_layer_units+[self.num_labels],
        )

        # 8. output Dropout层
        l8_dropout = Dropout(p=self.output_dropout_rate)(l6_full_connected_layer)
        # 9. softmax 分类层
        l9_softmax_output = Activation("softmax")(l8_dropout)
        model = Model(input=[l1_input], output=[l9_softmax_output])

        # softmax层的输出
        self.model_output = K.function([l1_input, K.learning_phase()], [l9_softmax_output])
        # 卷积层的输出，可以作为深度特征
        # self.conv1_feature_output = K.function([l1_input, K.learning_phase()], [l6_flatten])

        if self.verbose > 0:
            model.summary()

        return model



    def print_model_descibe(self):
        import pprint
        detail = {'rand_seed': self.rand_seed,
                  'verbose': self.verbose,
                  'optimizers': self.optimizers,
                  'input_dim': self.feature_encoder.vocabulary_size,
                  'input_length': self.input_length,
                  'num_labels': self.num_labels,
                  'conv1_filter_type': self.conv1_filter_type,
                  'conv2_filter_type': self.conv2_filter_type,
                  'output_dropout_rate': self.output_dropout_rate,
                  'nb_epoch': self.nb_epoch,
                  'earlyStoping_patience': self.earlyStoping_patience,
                  'lr':self.lr,
                  'batch_size':self.batch_size,
                  }
        pprint.pprint(detail)
        logging.debug(detail)
        return detail


if __name__ == '__main__':
    # 使用样例
    train_X = ['你好', '无聊', '测试句子', '今天天气不错', '我要买手机']
    trian_y = [1, 3, 2, 2, 3]
    test_X = ['句子', '你好', '你妹']
    test_y = [2, 3, 0]
    sentence_padding_length = 8
    feature_encoder = FeatureEncoder(
        sentence_padding_length=sentence_padding_length,
        verbose=0,
        need_segmented=True,
        full_mode=True,
        replace_number=True,
        remove_stopword=True,
        lowercase=True,
        padding_mode='left',
        add_unkown_word=True,
        mask_zero=True,
        to_onehot_array = True,
    )
    train_X_feature = feature_encoder.fit_transform(train_X)
    test_X_feature = feature_encoder.transform(test_X)
    # print train_X_feature.shape
    # print map(feature_encoder.transform_sentence, test_X)
    # quit()
    onehot_cnn = OnehotCNN(
        rand_seed=1377,
        verbose=1,
        feature_encoder=feature_encoder,
        # optimizers='adadelta',
        optimizers='sgd',
        input_length=sentence_padding_length,
        num_labels=5,
        conv1_filter_type=[[4, 2, -1, 'valid',(0,1)],
                          [4, 4, -1, 'valid',(0,1)],
                          [4, 5, -1, 'valid',(0,1)],
                          ],
        conv2_filter_type=[[16, 2, -1, 'valid',(2,1)]],
        full_connected_layer_units=[50],
        embedding_dropout_rate=0.,
        output_dropout_rate=0.,
        nb_epoch=100,
        nb_batch=5,
        earlyStoping_patience=20,
        lr=1e-1,
    )
    onehot_cnn.print_model_descibe()
    # 训练模型
    # 从保存的pickle中加载模型
    # onehot_cnn.model_from_pickle('model/modelA.pkl')
    onehot_cnn.fit((train_X_feature, trian_y),
                   (test_X_feature, test_y))
    onehot_cnn.accuracy((train_X_feature, trian_y), transform_input=False)

    print onehot_cnn.batch_predict(test_X_feature, transform_input=False)
    print onehot_cnn.batch_predict_bestn(test_X_feature, transform_input=False, bestn=2)
    print onehot_cnn.batch_predict(test_X, transform_input=True)
    print onehot_cnn.predict(test_X[0], transform_input=True)
    onehot_cnn.accuracy((test_X, test_y), transform_input=True)
    # 保存模型
    # onehot_cnn.save_model('model/modelA.pkl')

    print onehot_cnn.predict('你好吗', transform_input=True)
