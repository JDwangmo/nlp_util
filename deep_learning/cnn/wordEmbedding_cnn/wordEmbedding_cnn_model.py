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



class WordEmbeddingCNN(CnnBaseClass):
    '''
        一层CNN模型,随机初始化词向量,CNN-rand模型.借助Keras和jieba实现。
        架构各个层次分别为: 输入层,embedding层,dropout层,卷积层,1-max pooling层,全连接层,softmax层
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
                 input_dim=None,
                 word_embedding_dim=None,
                 embedding_init_weight=None,
                 embedding_weight_trainable = True,
                 input_length=None,
                 num_labels=None,
                 l1_conv_filter_type=None,
                 l2_conv_filter_type=None,
                 embedding_dropout_rate=0.5,
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
        )
        self.input_dim = input_dim
        self.word_embedding_dim = word_embedding_dim
        self.embedding_init_weight = embedding_init_weight
        self.input_length = input_length

        self.embedding_weight_trainable = embedding_weight_trainable

        self.l1_conv_filter_type = l1_conv_filter_type
        self.l2_conv_filter_type = l2_conv_filter_type
        self.full_connected_layer_units = full_connected_layer_units
        self.embedding_dropout_rate = embedding_dropout_rate
        self.kwargs = kwargs

        # 嵌入层的输出
        self.embedding_layer_output = None
        # 最后一层隐含层（倒数第二层）的输出
        self.last_hidden_layer = None
        self.conv1_feature_output = None

        # 构建模型
        self.build_model()

    def create_network(self):
        '''
            1. 创建 CNN 网络

                1. 输入层，2D，（n_batch,input_length）
                2. Embedding层,3D,（n_batch,input_length,embedding_dim）
                3. 输入dropout层，对Embedding层进行dropout.3D.
                4. Reshape层： 将embedding转换4-dim的shape，4D
                5. 第一层多size卷积层（含1-max pooling），使用三种size.
                6. Flatten层： 卷积的结果进行拼接,变成一列隐含层
                7. 全连接层
                8. softmax 分类层
            2. compile模型

        :return: cnn model network
        '''

        from keras.layers import Embedding, Input, Activation, Reshape, Dropout, Dense, Flatten
        from keras.models import Model
        from keras import backend as K

        # 1. 输入层
        l1_input_shape = ( self.input_length,)
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
        l4_reshape= Reshape((1, self.input_length, self.word_embedding_dim))(l3_dropout)
        # 5. 第一层卷积层：多size卷积层（含1-max pooling），使用三种size.
        l5_cnn = self.create_convolution_layer(
            input_layer=l4_reshape,
            convolution_filter_type=self.l1_conv_filter_type,
        )
        self.embedding_layer_output = Model(input=l1_input, output=[l5_cnn])
        # print (self.embedding_layer_output.get_weights())
        # model = Model(input=l1_input, output=[l5_cnn])
        # model.summary()
        # quit()

        # 6. 第二层卷积层：单size卷积层 和 max pooling 层
        l6_conv = self.create_convolution_layer(
            input_layer=l5_cnn,
            convolution_filter_type=self.l2_conv_filter_type,
        )

        # 6. Flatten层： 卷积的结果进行拼接,变成一列隐含层
        l6_flatten = Flatten()(l6_conv)

        # 7. 全连接层
        l7_full_connected_layer = self.create_full_connected_layer(
            input_layer=l6_flatten,
            units=self.full_connected_layer_units+[[self.num_labels,0.,'none','none']],
        )

        # 8. softmax 分类层
        l8_softmax_output = Activation("softmax")(l7_full_connected_layer)
        model = Model(input=[l1_input], output=[l8_softmax_output])

        # 最后一层隐含层（倒数第二层）的输出
        self.last_hidden_layer= K.function([l1_input, K.learning_phase()], [l7_full_connected_layer])
        # 最后输出层
        self.model_output = K.function([l1_input, K.learning_phase()], [l8_softmax_output])
        # 卷积层的输出，可以作为深度特征
        self.conv1_feature_output = K.function([l1_input, K.learning_phase()], [l6_flatten])

        if self.verbose > 0:
            model.summary()

        return model

    @staticmethod
    def cross_validation(cv_data, test_data, result_file_path, **kwargs):
        '''
            进行参数的交叉验证

        :param cv_data: k份训练数据
        :type cv_data: array-like
        :param test_data: 测试数据
        :type test_data: array-like
        :return:
        '''

        nb_epoch = kwargs['nb_epoch']
        verbose = kwargs['verbose']
        num_labels = 24
        feature_type = kwargs['feature_type']
        full_mode = kwargs['full_mode']
        rand_seed = kwargs['rand_seed']
        embedding_weight_trainable = kwargs['embedding_weight_trainable']
        l1_conv_filter_type = kwargs['l1_conv_filter_type']
        l2_conv_filter_type = kwargs['l2_conv_filter_type']
        k = kwargs['k']
        kwargs['to_embedding_weight'] = True
        word_embedding_dim = kwargs['word_embedding_dim']
        sentence_padding_length = kwargs['sentence_padding_length']
        word2vec_model_file_path = kwargs['word2vec_model_file_path']


        # 详细结果保存到...
        fout = open(result_file_path, 'w')
        print('=' * 150)
        print('调节的参数....')
        print('layer1:%s'%str(kwargs['layer1']))
        print('layer2:%s'%str(kwargs['layer2']))
        print('hidden1:%s'%str(kwargs['hidden1']))
        print('hidden2:%s'%str(kwargs['hidden2']))

        print(word2vec_model_file_path)
        print('embedding_weight_trainable:%s'%embedding_weight_trainable)
        print('feature_type:%s\nnb_epoch:%d\nrand_seed:%d' % (feature_type, nb_epoch, rand_seed))
        print('sentence_padding_length:%d\nword_embedding_dim:%d'%(sentence_padding_length,word_embedding_dim))
        print('l1_conv_filter_type:%s' % l1_conv_filter_type)
        print('l2_conv_filter_type:%s' % l2_conv_filter_type)
        print('k:%s' % k)
        print('=' * 150)

        fout.write('=' * 150 + '\n')
        fout.write('single单通道CNN-embedding cv结果:\n')
        fout.write('feature_type:%s\nnb_epoch:%d\nrand_seed:%d\n' % (feature_type, nb_epoch, rand_seed))
        fout.write('l1_conv_filter_type:%s\n' % l1_conv_filter_type)
        fout.write('l2_conv_filter_type:%s\n' % l2_conv_filter_type)
        fout.write('k:%s\n' % k)
        fout.write('=' * 150 + '\n')

        from data_processing_util.feature_encoder.onehot_feature_encoder import FeatureEncoder
        from data_processing_util.cross_validation_util import transform_cv_data

        feature_encoder = FeatureEncoder(
            sentence_padding_length=sentence_padding_length,
            verbose=1,
            need_segmented=True,
            full_mode=full_mode,
            remove_stopword=True,
            replace_number=True,
            lowercase=True,
            zhs2zht=True,
            remove_url=True,
            padding_mode='center',
            add_unkown_word=True,
            feature_type=feature_type,
        )

        all_cv_data = transform_cv_data(feature_encoder,
                                        cv_data,
                                        test_data,
                                        **kwargs
                                        )


        for layer1 in kwargs['layer1']:
            for layer2 in kwargs['layer2']:
                for hidden1 in kwargs['hidden1']:
                    for hidden2 in kwargs['hidden2']:

                        print('layer1:%d,layer2:%d,hidden1:%d,hidden2:%d' % (layer1, layer2, hidden1, hidden2))

                        fout.write('=' * 150 + '\n')
                        fout.write('layer1:%d,layer2:%d,hidden1:%d,hidden2:%d\n' % (layer1,
                                                                                    layer2,
                                                                                    hidden1,
                                                                                    hidden2
                                                                                    ))
                        # 五折
                        print('K折交叉验证开始...')
                        counter = 0
                        ave_acc = []
                        for dev_X, dev_y, val_X, val_y,init_weight in all_cv_data:
                            # print(dev_X.shape)
                            print('-' * 80)
                            fout.write('-' * 80 + '\n')
                            if counter == 0:
                                # 第一个数据是训练，之后是交叉验证
                                print('训练:')
                                fout.write('训练\n')
                            else:
                                print('第%d个验证' % counter)
                                fout.write('第%d个验证\n' % counter)


                            word_embedding_cnn = WordEmbeddingCNN(
                                rand_seed=rand_seed,
                                verbose=verbose,
                                feature_encoder=None,
                                # optimizers='adadelta',
                                optimizers='sgd',
                                input_dim=init_weight.shape[0],
                                word_embedding_dim=word_embedding_dim,
                                embedding_init_weight=init_weight,
                                embedding_weight_trainable=embedding_weight_trainable,
                                input_length=sentence_padding_length,
                                num_labels=num_labels,
                                l1_conv_filter_type=[
                                    [layer1, 3, -1, 'valid', [2, 1], 0., 'relu', 'none'],
                                     [layer1, 4, -1, 'valid', [2, 1], 0., 'relu', 'none'],
                                     [layer1, 5, -1, 'valid', [2, 1], 0., 'relu', 'none'],
                                     ],
                                l2_conv_filter_type=[
                                    # [layer2, l2_conv_filter_type[0], 1, 'valid', (k[1], 1), 0.25]
                                ],
                                full_connected_layer_units=[
                                    [hidden1, 0.5, 'none', 'none'],
                                    # [hidden2,0.5]
                                ],
                                embedding_dropout_rate=0.,
                                nb_epoch=nb_epoch,
                                nb_batch=32,
                                earlyStoping_patience=200,
                                lr=1e-2,
                            )
                            if verbose>0:
                                word_embedding_cnn.print_model_descibe()

                            # print (word_embedding_cnn.embedding_layer_output.get_weights()[0][1])
                            dev_loss, dev_accuracy, \
                            val_loss, val_accuracy = word_embedding_cnn.fit((dev_X, dev_y), (val_X, val_y))
                            # print (word_embedding_cnn.embedding_layer_output.get_weights()[0][1])
                            print('dev:%f,%f' % (dev_loss, dev_accuracy))
                            print('val:%f,%f' % (val_loss, val_accuracy))
                            fout.write('dev:%f,%f\n' % (dev_loss, dev_accuracy))
                            fout.write('val:%f,%f\n' % (val_loss, val_accuracy))
                            ave_acc.append(val_accuracy)
                            counter += 1

                        print('k折验证结果：%s' % ave_acc)
                        print('验证中平均准确率：%f' % np.average(ave_acc[1:]))
                        print('-' * 80)

                        fout.write('k折验证结果：%s\n' % ave_acc)
                        fout.write('平均：%f\n' % np.average(ave_acc[1:]))
                        fout.write('-' * 80 + '\n')
                        fout.flush()
        fout.close()

    def get_conv1_feature(self, sentence, transform_input=False):
        '''
            encoding,将句子以conv1的输出为编码

        :param sentence: 一个测试句子,
        :type sentence: array-like
        :param transform: 是否转换句子，如果为True,输入原始字符串句子即可，内部已实现转换成字典索引的形式。
        :type transform: array-like
        '''

        if transform_input:
            sentence = self.transform(sentence)

        conv1_feature = self.conv1_feature_output([np.asarray(sentence).reshape(1, -1), 0])[0]

        conv1_feature = conv1_feature.flatten()

        # -------------- print start : just print info -------------
        if self.verbose > 2:
            logging.debug('句子表示成%d维的特征' % (conv1_feature.shape))
            print('句子表示成%d维的特征' % (len(conv1_feature)))

        # -------------- print end : just print info -------------
        return conv1_feature




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
                  'lr':self.lr,
                  'batch_size':self.batch_size,
                  }
        pprint.pprint(detail)
        logging.debug(detail)
        return detail


def test_static_w2v():
    '''
        测试 CNN(static-w2v)

    :return:
    '''

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
    )
    train_X_feature = feature_encoder.fit_transform(train_X)
    test_X_feature = feature_encoder.transform(test_X)
    # print feature_encoder.train_padding_index
    # print map(feature_encoder.transform_sentence, test_X)
    # quit()
    word_embedding_dim = 50
    rand_embedding_cnn = WordEmbeddingCNN(
        rand_seed=1377,
        verbose=1,
        feature_encoder=feature_encoder,
        # optimizers='adadelta',
        optimizers='sgd',
        input_dim=feature_encoder.vocabulary_size ,
        word_embedding_dim=word_embedding_dim,
        # 设置embedding使用训练好的w2v模型初始化
        embedding_init_weight=feature_encoder.to_embedding_weight(
            '/home/jdwang/PycharmProjects/corprocessor/word2vec/vector/50dim/ood_sentence_vector1191_50dim.gem'),
        # 设置为训练时embedding层权重不变
        embedding_weight_trainable=False,
        input_length=sentence_padding_length,
        num_labels=5,
        l1_conv_filter_type=[
            [4, 2, word_embedding_dim, 'valid', (2, 1), 0.5],
            [4, 4, word_embedding_dim, 'valid', (2, 1), 0.],
            [4, 5, word_embedding_dim, 'valid', (2, 1), 0.],
        ],
        l2_conv_filter_type=[
            [16, 2, 1, 'valid', (2, 1), 0.]
        ],
        full_connected_layer_units=[50],
        embedding_dropout_rate=0.,
        nb_epoch=30,
        nb_batch=5,
        earlyStoping_patience=20,
        lr=1e-1,
    )
    rand_embedding_cnn.print_model_descibe()
    # 训练模型
    # 从保存的pickle中加载模型
    print (rand_embedding_cnn.embedding_layer_output.get_weights()[0][1])
    rand_embedding_cnn.fit((train_X_feature, trian_y),
                           (test_X_feature, test_y))
    print (rand_embedding_cnn.embedding_layer_output.get_weights()[0][1])
    rand_embedding_cnn.accuracy((train_X_feature, trian_y), transform_input=False)

    quit()
    print rand_embedding_cnn.batch_predict(test_X_feature, transform_input=False)
    print rand_embedding_cnn.batch_predict_bestn(test_X_feature, transform_input=False, bestn=2)
    print rand_embedding_cnn.batch_predict(test_X, transform_input=True)
    print rand_embedding_cnn.predict(test_X[0], transform_input=True)
    rand_embedding_cnn.get_conv1_feature(test_X_feature)
    rand_embedding_cnn.accuracy((test_X, test_y), transform_input=True)
    # 保存模型
    # onehot_cnn.save_model('model/modelA.pkl')

    print rand_embedding_cnn.predict('你好吗', transform_input=True)

def test_nonstatic_w2v():
    '''
        测试 CNN(nonstatic-w2v)

    :return:
    '''

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
    )
    train_X_feature = feature_encoder.fit_transform(train_X)
    test_X_feature = feature_encoder.transform(test_X)
    # print feature_encoder.train_padding_index
    # print map(feature_encoder.transform_sentence, test_X)
    # quit()
    word_embedding_dim = 50
    rand_embedding_cnn = WordEmbeddingCNN(
        rand_seed=1377,
        verbose=1,
        feature_encoder=feature_encoder,
        # optimizers='adadelta',
        optimizers='sgd',
        input_dim=feature_encoder.vocabulary_size ,
        word_embedding_dim=word_embedding_dim,
        # 设置embedding使用训练好的w2v模型初始化
        embedding_init_weight=feature_encoder.to_embedding_weight(
            '/home/jdwang/PycharmProjects/corprocessor/word2vec/vector/50dim/ood_sentence_vector1191_50dim.gem'),
        # 设置为训练时embedding层权重一起变化
        embedding_weight_trainable=True,
        input_length=sentence_padding_length,
        num_labels=5,
        l1_conv_filter_type=[
            [4, 2, word_embedding_dim, 'valid', (2, 1), 0.5],
            [4, 4, word_embedding_dim, 'valid', (2, 1), 0.],
            [4, 5, word_embedding_dim, 'valid', (2, 1), 0.],
        ],
        l2_conv_filter_type=[
            [16, 2, 1, 'valid', (2, 1), 0.]
        ],
        full_connected_layer_units=[50],
        embedding_dropout_rate=0.,
        nb_epoch=30,
        nb_batch=5,
        earlyStoping_patience=20,
        lr=1e-1,
    )
    rand_embedding_cnn.print_model_descibe()
    # 训练模型
    # 从保存的pickle中加载模型
    print (rand_embedding_cnn.embedding_layer_output.get_weights()[0][1])
    rand_embedding_cnn.fit((train_X_feature, trian_y),
                           (test_X_feature, test_y))
    print (rand_embedding_cnn.embedding_layer_output.get_weights()[0][1])
    rand_embedding_cnn.accuracy((train_X_feature, trian_y), transform_input=False)

    quit()
    print rand_embedding_cnn.batch_predict(test_X_feature, transform_input=False)
    print rand_embedding_cnn.batch_predict_bestn(test_X_feature, transform_input=False, bestn=2)
    print rand_embedding_cnn.batch_predict(test_X, transform_input=True)
    print rand_embedding_cnn.predict(test_X[0], transform_input=True)
    rand_embedding_cnn.get_conv1_feature(test_X_feature)
    rand_embedding_cnn.accuracy((test_X, test_y), transform_input=True)
    # 保存模型
    # onehot_cnn.save_model('model/modelA.pkl')

    print rand_embedding_cnn.predict('你好吗', transform_input=True)

if __name__ == '__main__':
    # 使用样例
    train_X = ['你好', '无聊', '测试句子', '今天天气不错', '我要买手机']
    trian_y = [1, 3, 2, 2, 3]
    test_X = ['句子', '你好', '你妹']
    test_y = [2, 3, 0]
    # test_static_w2v()
    test_nonstatic_w2v()

