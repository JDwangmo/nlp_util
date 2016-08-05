# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-06-23'
    Email:   '383287471@qq.com'
    Describe: CNN(seq)模型，以 onehot向量输入，之后搭建卷积层和隐含层
            用于实现两种类型的模型：
                - CNN（seq）: example/one_conv_layer_onehot_cnn.py
                - CNN(BOW): example/one_conv_layer_onehot_bow_cnn.py
            更多可参考： https://github.com/JDwangmo/coprocessor/tree/master/reference#3effective-use-of-word-order-for-text-categorization-with-convolutional-neural-networks
"""

import numpy as np
from deep_learning.cnn.common import CnnBaseClass
import logging
from data_processing_util.feature_encoder.onehot_feature_encoder import FeatureEncoder
import pprint


class OnehotBowCNN(CnnBaseClass):
    """
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
            10. get_feature_encoder: 静态方法，获取模型的特征编码器
    """

    def __init__(self,
                 rand_seed=1337,
                 verbose=0,
                 feature_encoder=None,
                 full_connected_layer_units=None,
                 optimizers='sgd',
                 input_length=None,
                 input_dim = None,
                 num_labels=None,
                 l1_conv_filter_type=None,
                 l2_conv_filter_type=None,
                 nb_epoch=10,
                 batch_size=32,
                 earlyStoping_patience=50,
                 **kwargs
                 ):
        """
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
        :param input_dim: cnn设置选项,onehot array dim .
        :type input_dim: int
        :param num_labels: cnn设置选项,最后输出层的大小,即分类类别的个数.
        :type num_labels: int
        :param l1_conv_filter_type: cnn设置选项,卷积层的类型.

            for example:每个列表代表一种类型(size)的卷积核,
                l1_conv_filter_type = [[100,2,word_embedding_dim,'valid',(1,1)],
                                    [100,4,word_embedding_dim,'valid',(1,1)],
                                    [100,6,word_embedding_dim,'valid',(1,1)],
                                   ]

        :type l1_conv_filter_type: array([])
        :param nb_epoch: cnn设置选项,cnn迭代的次数.
        :type nb_epoch: int
        :param earlyStoping_patience: cnn设置选项,earlyStoping的设置,如果迭代次数超过这个耐心值,依旧不下降,则stop.
        :type earlyStoping_patience: int
        """

        CnnBaseClass.__init__(
            self,
            rand_seed=rand_seed,
            verbose=verbose,
            feature_encoder=feature_encoder,
            optimizers=optimizers,
            input_length=input_length,
            num_labels=num_labels,
            nb_epoch=nb_epoch,
            batch_size= batch_size,
            earlyStoping_patience=earlyStoping_patience,
        )

        if feature_encoder != None:
            # 如果feature encoder 不为空，直接用 feature_encoder获取 长度和维度
            self.input_dim = feature_encoder.vocabulary_size
            self.input_length = feature_encoder.sentence_padding_length

        else:
            self.input_dim = input_dim
            self.input_length = input_length

        self.l1_conv_filter_type = l1_conv_filter_type
        self.l2_conv_filter_type = l2_conv_filter_type
        self.full_connected_layer_units = full_connected_layer_units
        self.kwargs = kwargs

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

        from keras.layers import Input, Activation, Reshape, Dropout, Flatten,BatchNormalization
        from keras.models import Model
        # from keras import backend as K

        # 1. 输入层
        l1_input_shape = ( self.input_length,self.input_dim)
        # l1_input_shape = ( None,self.input_dim)
        l1_input = Input(shape=l1_input_shape)

        # 2. Reshape层： 将embedding转换4-dim的shape
        l2_reshape_output_shape = (1, l1_input_shape[0], l1_input_shape[1])
        # print(l2_reshape_output_shape)
        # quit()
        l2_reshape= Reshape(l2_reshape_output_shape)(l1_input)
        # l2_reshape = BatchNormalization(axis=1)(l2_reshape)

        # 3. 第一层卷积层：多size卷积层（含1-max pooling），使用三种size.
        l3_conv = self.create_convolution_layer(
            input_layer=l2_reshape,
            convolution_filter_type=self.l1_conv_filter_type,
        )
        # 4. 第二层卷积层：单size卷积层 和 max pooling 层
        l4_conv = self.create_convolution_layer(
            input_layer=l3_conv,
            convolution_filter_type=self.l2_conv_filter_type,
        )
        # model = Model(input=l1_input, output=[l3_conv])
        # model.summary()
        # quit()
        # 5. Flatten层： 卷积的结果进行拼接,变成一列隐含层
        l5_flatten = Flatten()(l4_conv)
        # 6. 全连接层
        l6_full_connected_layer = self.create_full_connected_layer(
            input_layer=l5_flatten,
            units=self.full_connected_layer_units
        )

        l7_output_layer = self.create_full_connected_layer(
            input_layer=l6_full_connected_layer,
            units=[[self.num_labels, 0., 'none', 'none']]
        )

        # 8. softmax分类层
        l8_softmax_output = Activation("softmax")(l7_output_layer)

        model = Model(input=l1_input, output=[l8_softmax_output])

        if self.verbose > 0:
            model.summary()

        return model

    @staticmethod
    def get_feature_encoder(**kwargs):
        '''
            返回 该模型的输入 特征编码器

        :param kwargs: 可设置参数 [ sentence_padding_length(*), full_mode(#,False), feature_type(#,word),verbose(#,0)],word2vec_to_solve_oov[#,False],word2vec_model_file_path[#,None],加*表示必须提供，加#表示可选，不写则默认。

        :return:
        '''

        assert kwargs.has_key('input_length'),'请提供 input_length 的属性值'

        feature_encoder = FeatureEncoder(
            sentence_padding_length=kwargs['input_length'],
            verbose=kwargs.get('verbose',0),
            need_segmented=True,
            full_mode=kwargs.get('full_mode',False),
            replace_number=True,
            remove_stopword=True,
            lowercase=True,
            padding_mode='left',
            add_unkown_word=True,
            feature_type=kwargs.get('feature_type','word'),
            zhs2zht=True,
            remove_url=True,
            # 设置为True，输出 onehot array
            to_onehot_array=True,
            word2vec_to_solve_oov=kwargs.get('word2vec_to_solve_oov',False),
            word2vec_model_file_path=kwargs.get('word2vec_model_file_path',None)
        )
        if kwargs.get('verbose',0)>0:
            pprint.pprint(kwargs)

        return feature_encoder

    def print_model_descibe(self):
        detail = {'rand_seed': self.rand_seed,
                  'verbose': self.verbose,
                  'optimizers': self.optimizers,
                  'input_dim': self.feature_encoder.vocabulary_size,
                  'input_length': self.input_length,
                  'num_labels': self.num_labels,
                  'l1_conv_filter_type': self.l1_conv_filter_type,
                  'l2_conv_filter_type': self.l2_conv_filter_type,
                  'full_connected_layer_units':self.full_connected_layer_units,
                  'nb_epoch': self.nb_epoch,
                  'earlyStoping_patience': self.earlyStoping_patience,
                  'lr':self.lr,
                  'batch_size':self.batch_size,
                  }
        pprint.pprint(detail)
        logging.debug(detail)
        return detail

def test_onehot_bow_cnn():
    # 使用样例
    train_X = ['你好', '无聊', '测试句子', '今天天气不错', '我要买手机']
    trian_y = [1, 3, 2, 2, 3]
    test_X = ['句子', '你好', '你妹']
    test_y = [2, 3, 0]
    sentence_padding_length = 8
    feature_encoder = OnehotBowCNN.get_feature_encoder(
        input_length=sentence_padding_length,
        verbose=1,
        feature_type='word',
    )

    train_X_feature = feature_encoder.fit_transform(train_X)
    test_X_feature = feature_encoder.transform(test_X)
    print(','.join(feature_encoder.vocabulary))
    print train_X_feature.shape
    print train_X_feature

    # quit()
    onehot_cnn = OnehotBowCNN(
        rand_seed=1377,
        verbose=1,
        feature_encoder=feature_encoder,
        # optimizers='adadelta',
        optimizers='sgd',
        input_length=sentence_padding_length,
        input_dim=feature_encoder.vocabulary_size,
        num_labels=5,
        l1_conv_filter_type=[
            # [5, 3, -1, 'valid', (2, 1), 0.5, 'relu', 'none'],
            [5, 2, -1, 'bow', (2, 1), 0.5, 'relu', 'none'],
            # [5, 6, 1, 'valid', (-2, 1), 0.],
        ],
        l2_conv_filter_type=[
            # [16, 2, -1, 'valid',(2,1),0.5, 'relu', 'none']
        ],
        full_connected_layer_units=[
            # (50, 0.5, 'relu', 'none'),
        ],
        embedding_dropout_rate=0.5,
        nb_epoch=30,
        nb_batch=5,
        earlyStoping_patience=20,
        lr=1e-2,
    )
    onehot_cnn.print_model_descibe()
    # 训练模型
    # 从保存的pickle中加载模型
    # onehot_cnn.model_from_pickle('model/modelA.pkl')
    onehot_cnn.fit((train_X_feature, trian_y),
                   (test_X_feature, test_y))
    print(trian_y)
    # loss, train_accuracy = onehot_cnn.model.evaluate(train_X_feature, trian_y)

    onehot_cnn.accuracy((train_X_feature, trian_y), transform_input=False)
    quit()
    print onehot_cnn.batch_predict(test_X_feature, transform_input=False)
    print onehot_cnn.batch_predict_bestn(test_X_feature, transform_input=False, bestn=2)
    print onehot_cnn.batch_predict(test_X, transform_input=True)
    print onehot_cnn.predict(test_X[0], transform_input=True)
    onehot_cnn.accuracy((test_X, test_y), transform_input=True)
    # 保存模型
    # onehot_cnn.save_model('model/modelA.pkl')

    print onehot_cnn.predict('你好吗', transform_input=True)

if __name__ == '__main__':
    test_onehot_bow_cnn()
    # test_onehot_bow_cnn_cv()