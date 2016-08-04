# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-08-03'
    Email:   '383287471@qq.com'
    Describe:
"""
from deep_learning.cnn.common import CnnBaseClass
import logging
class DCNN(CnnBaseClass):

    def __init__(self,
                 rand_seed=1337,
                 verbose=0,
                 feature_encoder=None,
                 full_connected_layer_units=None,
                 optimizers='sgd',
                 input_dim=None,
                 word_embedding_dim=None,
                 embedding_init_weight=None,
                 embedding_weight_trainable=True,
                 input_length=None,
                 num_labels=None,
                 l1_conv_filter_type=None,
                 l2_conv_filter_type=None,
                 embedding_dropout_rate=0.,
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
        :param input_dim: embedding层输入(onehot)的维度,即 字典大小, 0 给填充用
        :type input_dim: int
        :param word_embedding_dim: cnn设置选项,embedding层词向量的维度(长度).
        :type word_embedding_dim: int
        :param embedding_init_weight: cnn设置选项,embedding层词向量的权重初始化方式,有2种,.
            1. None: 使用随机初始化权重.
            2. 不是None：若有提供权重，则使用训练好的词向量进行初始化.
        :type embedding_init_weight: 2d array-like
        :param embedding_weight_trainable: 设置embedding层的权重是否 static or nonstatic
        :type embedding_weight_trainable: bool
        :param input_length: cnn设置选项,输入句子(序列)的长度.
        :type input_length: int
        :param num_labels: cnn设置选项,最后输出层的大小,即分类类别的个数.
        :type num_labels: int
        :param l1_conv_filter_type: cnn设置选项,卷积层的类型.

            for example:每个列表代表一种类型(size)的卷积核,
                l1_conv_filter_type = [[100,2,word_embedding_dim,'valid',(1,1)],
                                    [100,4,word_embedding_dim,'valid',(1,1)],
                                    [100,6,word_embedding_dim,'valid',(1,1)],
                                   ]

        :type l1_conv_filter_type: array-like
        :param k: cnn设置选项,k-max pooling层的的k值,即设置要获取 前k个 值 ,默认为 1-max
        :type k: int
        :param embedding_dropout_rate: cnn设置选项,dropout层的的dropout rate,对embedding层进入dropuout,如果为0,则不dropout
        :type embedding_dropout_rate: float
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
            **kwargs
        )
        self.word_embedding_dim = word_embedding_dim
        self.embedding_init_weight = embedding_init_weight

        self.embedding_weight_trainable = embedding_weight_trainable
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
        self.embedding_dropout_rate = embedding_dropout_rate
        self.kwargs = kwargs

        # 嵌入层的输出
        self.embedding_layer_output = None

        # 构建模型
        self.build_model()

    def create_network(self):
        from keras.layers import Embedding, Input, Activation, Reshape, Dropout, Flatten, Dense
        from deep_learning.cnn.custom_layers import FoldingLayer
        from keras.models import Model
        from keras import backend as K

        # 1. 输入层
        l1_input_shape = (self.input_length,)
        l1_input = Input(shape=l1_input_shape, dtype='int32')

        # model_input = Input((self.input_length,), dtype='int32')
        # 2. Embedding层
        if self.embedding_init_weight is None:
            weight = None
        else:
            weight = [self.embedding_init_weight]
        l2_embedding = Embedding(
            input_dim=self.input_dim,
            output_dim=self.word_embedding_dim,
            input_length=self.input_length,
            # mask_zero = True,
            weights=weight,
            init='uniform',
            trainable=self.embedding_weight_trainable,
        )(l1_input)
        # 3. Dropout层，对Embedding层进行dropout
        # 输入dropout层,embedding_dropout_rate!=0,则对embedding增加doupout层
        if self.embedding_dropout_rate:
            l3_dropout = Dropout(p=self.embedding_dropout_rate)(l2_embedding)
        else:
            l3_dropout = l2_embedding
        # 4. Reshape层： 将embedding转换4-dim的shape
        l4_reshape = Reshape((1, self.input_length, self.word_embedding_dim))(l3_dropout)
        # 5. 第一层卷积层：多size卷积层（含1-max pooling），使用三种size.
        l5_cnn = self.create_convolution_layer(
            input_layer=l4_reshape,
            convolution_filter_type=self.l1_conv_filter_type,
        )
        # l5_cnn = FoldingLayer()(l5_cnn)

        # l5_cnn = Flatten()(l5_cnn)

        # model = Model(input=l1_input, output=[l5_cnn])
        # print (model.get_weights())


        # 6. 第二层卷积层：单size卷积层 和 max pooling 层
        l6_conv = self.create_convolution_layer(
            input_layer=l5_cnn,
            convolution_filter_type=self.l2_conv_filter_type,
        )
        # 折叠层
        l7_fold = FoldingLayer()(l6_conv)

        # 6. Flatten层： 卷积的结果进行拼接,变成一列隐含层
        l7_flatten = Flatten()(l7_fold)
        # l6_flatten= BatchNormalization(axis=1)(l6_flatten)
        # 7. 全连接层
        l8_full_connected_layer = self.create_full_connected_layer(
            input_layer=l7_flatten,
            units=self.full_connected_layer_units,
        )
        # l7_activation = Activation("relu")(l7_full_connected_layer)

        l9_output = self.create_full_connected_layer(
            input_layer=l8_full_connected_layer,
            units=[[self.num_labels, 0., 'none', 'none']],
        )

        # 8. softmax 分类层
        l8_softmax_output = Activation("softmax")(l9_output)

        model = Model(input=[l1_input], output=[l8_softmax_output])

        #

        # import numpy as np
        # data = np.random.randint(0, 2, 80).reshape((10, 8))
        # print(data.shape)
        # print(model.fit(data))
        # model.summary()
        # quit()
        if self.verbose > 0:
            model.summary()

        return model

    @staticmethod
    def get_feature_encoder(**kwargs):
        '''
            获取该分类器的特征编码器

        :param kwargs:  可设置参数 [ input_length(*), full_mode(#,False), feature_type(#,word),verbose(#,0)],加*表示必须提供，加#表示可选，不写则默认。
        :return:
        '''

        assert kwargs.has_key('input_length'), '请提供 input_length 的属性值'

        from data_processing_util.feature_encoder.onehot_feature_encoder import FeatureEncoder
        feature_encoder = FeatureEncoder(
            sentence_padding_length=kwargs['input_length'],
            verbose=kwargs.get('verbose', 0),
            need_segmented=True,
            full_mode=kwargs.get('full_mode', False),
            remove_stopword=True,
            replace_number=True,
            lowercase=True,
            zhs2zht=True,
            remove_url=True,
            padding_mode='center',
            add_unkown_word=True,
            feature_type=kwargs.get('feature_type', 'word')
        )

        return feature_encoder

    def print_model_descibe(self):
        import pprint
        detail = {'rand_seed': self.rand_seed,
                  'verbose': self.verbose,
                  'optimizers': self.optimizers,
                  'input_dim': self.input_dim,
                  'word_embedding_dim': self.word_embedding_dim,
                  'embedding_weight_trainable': self.embedding_weight_trainable,
                  'input_length': self.input_length,
                  'num_labels': self.num_labels,
                  'l1_conv_filter_type': self.l1_conv_filter_type,
                  'l2_conv_filter_type': self.l2_conv_filter_type,
                  'full_connected_layer_units': self.full_connected_layer_units,
                  'embedding_dropout_rate': self.embedding_dropout_rate,
                  'nb_epoch': self.nb_epoch,
                  'earlyStoping_patience': self.earlyStoping_patience,
                  'embedding_init use rand': self.embedding_init_weight is None,
                  'lr': self.lr,
                  'batch_size': self.batch_size,
                  }
        pprint.pprint(detail)
        logging.debug(detail)
        return detail

def test_dcnn():
    '''
        测试 CNN(static-w2v)

    :return:
    '''

    sentence_padding_length = 8
    feature_encoder = DCNN.get_feature_encoder(
        input_length=sentence_padding_length,
        verbose=0,
        full_mode=False,
        feature_type='word',
    )
    train_X_feature = feature_encoder.fit_transform(train_X)
    test_X_feature = feature_encoder.transform(test_X)
    # print feature_encoder.train_padding_index
    # print map(feature_encoder.transform_sentence, test_X)
    # quit()
    word_embedding_dim = 50
    static_w2v_cnn = DCNN(
        rand_seed=1377,
        verbose=1,
        feature_encoder=feature_encoder,
        # optimizers='adadelta',
        optimizers='sgd',
        input_dim=feature_encoder.vocabulary_size,
        word_embedding_dim=word_embedding_dim,
        # 设置embedding使用训练好的w2v模型初始化
        embedding_init_weight=None,
        # embedding_init_weight=feature_encoder.to_embedding_weight('/home/jdwang/PycharmProjects/corprocessor/word2vec/vector/50dim/ood_sentence_vector1191_50dim.gem'),
        # 设置为训练时embedding层权重变化
        embedding_weight_trainable=True,
        input_length=sentence_padding_length,
        num_labels=5,
        l1_conv_filter_type=[
            [4, 3, -1, '1D', (-4, 1), 0.,'none','none'],
            # [4, 3, -1, '1D', (-2, 1), 0., 'none', 'none'],
        ],
        l2_conv_filter_type=[
            # [2, 2, -1, '1D', (-3, 1), 0.,'none','none'],
        ],
        full_connected_layer_units=[
            # [50]
        ],
        embedding_dropout_rate=0.,
        nb_epoch=30,
        batch_size=5,
        earlyStoping_patience=30,
        lr=1e-2,
    )
    # quit()
    static_w2v_cnn.print_model_descibe()
    # 训练模型
    # 从保存的pickle中加载模型
    # print (static_w2v_cnn.embedding_layer_output.get_weights()[0][1])
    # import numpy as np
    # data = np.random.randint(0, 2, 80).reshape((10, 8))
    # print(data.shape)
    # print(static_w2v_cnn.model.predict(data).shape)
    # quit()
    static_w2v_cnn.fit((train_X_feature,trian_y),(test_X_feature, test_y))
    # print (static_w2v_cnn.embedding_layer_output.get_weights()[0][1])
    # static_w2v_cnn.accuracy((train_X_feature, trian_y), transform_input=False)

    quit()
    print static_w2v_cnn.batch_predict(test_X_feature, transform_input=False)
    print static_w2v_cnn.batch_predict_bestn(test_X_feature, transform_input=False, bestn=2)
    print static_w2v_cnn.batch_predict(test_X, transform_input=True)
    print static_w2v_cnn.predict(test_X[0], transform_input=True)
    static_w2v_cnn.get_conv1_feature(test_X_feature)
    static_w2v_cnn.accuracy((test_X, test_y), transform_input=True)
    # 保存模型
    # onehot_cnn.save_model('model/modelA.pkl')

    print static_w2v_cnn.predict('你好吗', transform_input=True)


if __name__ == '__main__':
    # 使用样例
    train_X = ['你好', '无聊', '测试句子', '今天天气不错', '我要买手机']
    trian_y = [1, 3, 2, 2, 3]
    test_X = ['句子', '你好', '你妹']
    test_y = [2, 3, 0]
    test_dcnn()
    # test_nonstatic_w2v()
