# encoding=utf8
__author__ = 'jdwang'
__date__ = 'create date: 2016-06-23'
__email__ = '383287471@qq.com'

import numpy as np
import pandas as pd
import logging
import cPickle as pickle
from data_processing_util.feature_encoder.onehot_feature_encoder import FeatureEncoder
import theano.tensor as T
from sklearn.metrics import f1_score

class RandEmbeddingCNN(object):
    '''
        一层CNN模型,随机初始化词向量,CNN-rand模型.借助Keras和jieba实现。
        架构各个层次分别为: 输入层,embedding层,dropout层,卷积层,1-max pooling层,全连接层,dropout层,softmax层
        具体见:
            https://github.com/JDwangmo/coprocessor#2convolutional-neural-networks-for-sentence-classification
    '''

    def __init__(self,
                 rand_seed=1337,
                 verbose=0,
                 input_dim=None,
                 word_embedding_dim=None,
                 input_length = None,
                 num_labels = None,
                 conv_filter_type = None,
                 k = 1,
                 embedding_dropout_rate = 0.5,
                 output_dropout_rate = 0.5,
                 nb_epoch=100,
                 earlyStoping_patience = 50,
                 ):
        '''
            1. 初始化参数
            2. 构建模型

        :param rand_seed: 随机种子,假如设置为为None时,则随机取随机种子
        :type rand_seed: int
        :param verbose: 数值越大,输出更详细的信息
        :type verbose: int
        :param input_dim: embedding层输入(onehot)的维度,即 字典大小+1,+1是为了留出0给填充用
        :type input_dim: int
        :param word_embedding_dim: cnn设置选项,embedding层词向量的维度(长度).
        :type word_embedding_dim: int
        :param input_length: cnn设置选项,输入句子(序列)的长度.
        :type input_length: int
        :param num_labels: cnn设置选项,最后输出层的大小,即分类类别的个数.
        :type num_labels: int
        :param conv_filter_type: cnn设置选项,卷积层的类型.

            for example:每个列表代表一种类型(size)的卷积核,
                conv_filter_type = [[100,2,word_embedding_dim,'valid'],
                                    [100,4,word_embedding_dim,'valid'],
                                    [100,6,word_embedding_dim,'valid'],
                                   ]

        :type conv_filter_type: array-like
        :param k: cnn设置选项,k-max pooling层的的k值,即设置要获取 前k个 值 ,默认为 1-max
        :type k: int
        :param embedding_dropout_rate: cnn设置选项,dropout层的的dropout rate,对embedding层进入dropuout,如果为0,则不dropout
        :type embedding_dropout_rate: float
        :param output_dropout_rate: cnn设置选项,dropout层的的dropout rate,对输出层进入dropuout,如果为0,则不dropout
        :type output_dropout_rate: float
        :param nb_epoch: cnn设置选项,cnn迭代的次数.
        :type nb_epoch: int
        :param earlyStoping_patience: cnn设置选项,earlyStoping的设置,如果迭代次数超过这个耐心值,依旧不下降,则stop.
        :type earlyStoping_patience: int
        '''

        self.verbose = verbose
        self.rand_seed = rand_seed
        self.verbose = verbose
        self.input_dim = input_dim
        self.word_embedding_dim = word_embedding_dim
        self.input_length = input_length
        self.num_labels = num_labels
        self.conv_filter_type = conv_filter_type
        self.k = k
        self.embedding_dropout_rate = embedding_dropout_rate
        self.output_dropout_rate = output_dropout_rate
        self.nb_epoch = nb_epoch
        self.earlyStoping_patience=earlyStoping_patience

        # cnn model
        self.model = None
        # cnn model 的输出函数
        self.model_output = None
        # 构建模型
        self.build_model()

    def kmaxpooling(self):
        '''
            分别定义 kmax 的output 和output shape
            !但是k-max的实现用到Lambda,而pickle无法dump function对象,所以使用该模型的时候,保存不了模型,待解决.
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
            input = T.transpose(input,axes=(0,1,3,2))
            sorted_values = T.argsort(input, axis=3)
            topmax_indexes = sorted_values[:, :, :, -self.k:]
            # sort indexes so that we keep the correct order within the sentence
            topmax_indexes_sorted = T.sort(topmax_indexes)

            # given that topmax only gives the index of the third dimension, we need to generate the other 3 dimensions
            dim0 = T.arange(0, input.shape[0]).repeat(input.shape[1] * input.shape[2] * self.k)
            dim1 = T.arange(0, input.shape[1]).repeat(self.k * input.shape[2]).reshape((1, -1)).repeat(input.shape[0],
                                                                                                  axis=0).flatten()
            dim2 = T.arange(0, input.shape[2]).repeat(self.k).reshape((1, -1)).repeat(input.shape[0] * input.shape[1],
                                                                                 axis=0).flatten()
            dim3 = topmax_indexes_sorted.flatten()
            return T.transpose(input[dim0, dim1, dim2, dim3].reshape((input.shape[0], input.shape[1],input.shape[2], self.k)),axes=(0,1,3,2))

        def kmaxpooling_output_shape(input_shape):
            return (input_shape[0], input_shape[1], self.k, input_shape[3])

        from keras.layers import Lambda
        return Lambda(kmaxpooling_output,kmaxpooling_output_shape,name='k-max')

    def build_model(self):
        '''
            构建CNN模型
                1. 构建第一层卷积层和1-max pooling
                2.将所有层连接起来
        :return:
        '''

        if self.rand_seed is not None:
            np.random.seed(self.rand_seed)

        from keras.layers import Embedding, Convolution2D, Input, Activation, MaxPooling2D, Reshape, Dropout, Dense, \
            Flatten, Merge
        from keras.models import Sequential, Model
        from keras import backend as K

        # -------------- region start : 1. 构建第一层卷积层和1-max pooling -------------

        if self.verbose > 1 :
            logging.debug('-' * 20)
            print '-' * 20
            logging.debug('1. 构建第一层卷积层和1-max pooling')
            print '1. 构建第一层卷积层和1-max pooling'
        # -------------- code start : 开始 -------------

        # 构建第一层卷积层和1-max pooling
        conv_layers = []
        for items in self.conv_filter_type:

            nb_filter, nb_row, nb_col, border_mode = items

            m = Sequential()
            m.add(Convolution2D(nb_filter,
                                nb_row,
                                nb_col,
                                border_mode=border_mode,
                                input_shape=(1,
                                             self.input_length,
                                             self.word_embedding_dim)
                                ))
            m.add(Activation('relu'))

            # 1-max
            if self.k == 1:
                if border_mode == 'valid':
                    pool_size = (self.input_length - nb_row + 1, 1)
                elif border_mode == 'same':
                    pool_size = (self.input_length, 1)
                m.add(MaxPooling2D(pool_size=pool_size, name='1-max'))
            else:
                # k-max pooling
                # todo
                # 因为kmax需要用到Lambda,而pickle无法dump function对象,所以使用该模型的时候,保存不了模型,待解决.
                m.add(self.kmaxpooling())
            # m.summary()
            conv_layers.append(m)

        # 卷积的结果进行拼接,变成一列隐含层
        cnn_model = Sequential()
        cnn_model.add(Merge(conv_layers, mode='concat', concat_axis=1))
        cnn_model.add(Flatten())
        # print cnn_model.summary()
        # quit()

        # -------------- code start : 结束 -------------
        if self.verbose > 1 :
            logging.debug('-' * 20)
            print '-' * 20
        # -------------- region end : 1. 构建第一层卷积层和1-max pooling ---------------

        # -------------- region start : 2.将所有层连接起来 -------------
        if self.verbose > 1 :
            logging.debug('-' * 20)
            print '-' * 20
            logging.debug('2.将所有层连接起来')
            print '2.将所有层连接起来'
        # -------------- code start : 开始 -------------

        # 输入层
        model_input = Input((self.input_length,), dtype='int64')
        # embedding层
        embedding = Embedding(input_dim=self.input_dim,
                              output_dim=self.word_embedding_dim,
                              input_length=self.input_length,
                              # mask_zero = True,
                              init='uniform'
                              )(model_input)
        # 输入dropout层,embedding_dropout_rate!=0,则对embedding增加doupout层
        if self.embedding_dropout_rate:
            embedding = Dropout(p=self.embedding_dropout_rate)(embedding)

        # 将embedding转换4-dim的shape
        embedding_4_dim = Reshape((1, self.input_length, self.word_embedding_dim))(embedding)
        #
        conv1_output = cnn_model([embedding_4_dim] * len(self.conv_filter_type))

        full_connected_layers = Dense(output_dim=self.num_labels, init="glorot_uniform", activation='relu')(
            conv1_output)

        dropout_layers = Dropout(p=self.output_dropout_rate)(full_connected_layers)

        softmax_output = Activation("softmax")(dropout_layers)

        self.model = Model(input=[model_input], output=[softmax_output])

        self.model_output = K.function([model_input, K.learning_phase()],[softmax_output])

        if self.verbose>1:
            self.model.summary()

        # -------------- code start : 结束 -------------
        if self.verbose > 1 :
            logging.debug('-' * 20)
            print '-' * 20
        # -------------- region end : 2.将所有层连接起来 ---------------

    def to_categorical(self,y):
        '''
        将y转成适合CNN的格式,即标签y展开成onehot编码,比如
            y = [1,2]--> y = [[0,1 ],[1,0]]
        :param y: 标签列表,比如: [1,1,2,3]
        :type y: array1D-like
        :return: y的onehot编码
        :rtype: array2D-like
        '''
        from keras.utils import np_utils
        y_onehot = np_utils.to_categorical(y,nb_classes=self.num_labels)
        return y_onehot

    def fit(self, train_data, validation_data):
        '''
            cnn model 的训练
                1. 设置优化算法,earlystop等
                2. 对数据进行格式转换,比如 转换 y 的格式:转成onehot编码
                3. 模型训练

        :param train_data: 训练数据,格式为:(train_X, train_y),train_X中每个句子以字典索引的形式表示(使用data_processing_util.feature_encoder.onehot_feature_encoder编码器编码),train_y是一个整形列表.
        :type train_data: (array-like,array-like)
        :param validation_data: 验证数据,格式为:(validation_X, validation_y),validation_X中每个句子以字典索引的形式表示(使用data_processing_util.feature_encoder.onehot_feature_encoder编码器编码),validation_y是一个整形列表.
        :type validation_data: (array-like,array-like)
        :return: None
        '''

        from keras.callbacks import EarlyStopping

        # -------------- region start : 1. 设置优化算法,earlystop等 -------------
        logging.debug('-' * 20)
        print '-' * 20
        if self.verbose > 1 :
            logging.debug('1. 设置优化算法,earlystop等')
            print '1. 设置优化算法,earlystop等'
        # -------------- code start : 开始 -------------

        # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

        self.model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        early_stop = EarlyStopping(patience=self.earlyStoping_patience, verbose=self.verbose)

        # -------------- code start : 结束 -------------
        if self.verbose > 1 :
            logging.debug('-' * 20)
            print '-' * 20
        # -------------- region end : 1. 设置优化算法,earlystop等 ---------------
        # -------------- region start : 2. 对数据进行格式转换,比如 转换 y 的格式:转成onehot编码 -------------
        if self.verbose > 1 :
            logging.debug('-' * 20)
            print '-' * 20
            logging.debug('2. 对数据进行格式转换,比如 转换 y 的格式:转成onehot编码')
            print '2. 对数据进行格式转换,比如 转换 y 的格式:转成onehot编码'
        # -------------- code start : 开始 -------------

        train_X, train_y = train_data
        train_X = np.asarray(train_X)
        validation_X,validation_y =validation_data
        validation_X = np.asarray(validation_X)

        train_y = self.to_categorical(train_y)

        validation_y = self.to_categorical(validation_y)

        # -------------- code start : 结束 -------------
        if self.verbose > 1 :
            logging.debug('-' * 20)
            print '-' * 20
        # -------------- region end : 2. 对数据进行格式转换,比如 转换 y 的格式:转成onehot编码 ---------------
        # -------------- region start : 3. 模型训练 -------------
        if self.verbose > 1 :
            logging.debug('-' * 20)
            print '-' * 20
            logging.debug('3. 模型训练')
            print '3. 模型训练'
        # -------------- code start : 开始 -------------
        self.model.fit(train_X,
                       train_y,
                       nb_epoch=self.nb_epoch,
                       verbose=self.verbose,
                       # validation_split=0.1,
                       validation_data=(validation_X,validation_y),
                       shuffle=True,
                       batch_size=32,
                       callbacks=[early_stop]
                       )

        # -------------- code start : 结束 -------------
        if self.verbose > 1 :
            logging.debug('-' * 20)
            print '-' * 20
        # -------------- region end : 3. 模型训练 ---------------

    def save_model(self,path):
        '''
            保存模型,保存成pickle形式
        :param path: 模型保存的路径
        :type path: 模型保存的路径
        :return:
        '''
        pickle.dump(self.model_output, open(path, 'wb'))

    def model_from_pickle(self,path):
        '''
            从模型文件中直接加载模型
        :param path:
        :return: RandEmbeddingCNN object
        '''
        self.model_output = pickle.load(file(path,'rb'))

    def predict(self,sentence_index):
        '''
            预测,对输入的句子进行预测

        :param sentence_index: 测试句子,以字典索引的形式
        :type sentence_index: array-like
        '''
        y_pred = self.model_output([np.asarray(sentence_index).reshape(1,-1),0])[0]
        y_pred = y_pred.argmax(axis=-1)[0]
        return y_pred

    def accuracy(self,test_data):
        '''
            预测,对输入的句子进行预测,并给出准确率
                1. 转换格式
                2. 批量预测
                3. 统计准确率等
                4. 统计F1(macro) :统计各个类别的F1值，然后进行平均

        :param sentence_index: 测试句子,以字典索引的形式
        :type sentence_index: array-like
        '''
        # -------------- region start : 1. 转换格式 -------------
        if self.verbose > 1 :
            logging.debug('-' * 20)
            print '-' * 20
            logging.debug('1. 转换格式')
            print '1. 转换格式'
        # -------------- code start : 开始 -------------

        test_X, test_y = test_data
        test_X = np.asarray(test_X)


        # -------------- code start : 结束 -------------
        if self.verbose > 1 :
            logging.debug('-' * 20)
            print '-' * 20
        # -------------- region end : 1. 转换格式 ---------------

        # -------------- region start : 2. 批量预测 -------------
        if self.verbose > 1 :
            logging.debug('-' * 20)
            print '-' * 20
            logging.debug('2. 批量预测')
            print '2. 批量预测'
        # -------------- code start : 开始 -------------

        y_pred = self.model_output([test_X,0])[0]
        y_pred = y_pred.argmax(axis=-1)

        # -------------- code start : 结束 -------------
        if self.verbose > 1 :
            logging.debug('-' * 20)
            print '-' * 20
        # -------------- region end : 2. 批量预测 ---------------

        # -------------- region start : 3 & 4. 计算准确率和F1值 -------------
        if self.verbose > 1 :
            logging.debug('-' * 20)
            print '-' * 20
            logging.debug('3 & 4. 计算准确率和F1值')
            print '3 & 4. 计算准确率和F1值'
        # -------------- code start : 开始 -------------

        is_correct = y_pred==test_y
        logging.debug('正确的个数:%d'%(sum(is_correct)))
        print '正确的个数:%d'%(sum(is_correct))
        accu = sum(is_correct)/(1.0*len(test_y))
        logging.debug('准确率为:%f'%(accu))
        print '准确率为:%f'%(accu)

        f1 = f1_score(test_y,y_pred.tolist(),average=None)
        logging.debug('F1为：%s'%(str(f1)))
        print 'F1为：%s'%(str(f1))

        # -------------- code start : 结束 -------------
        if self.verbose > 1 :
            logging.debug('-' * 20)
            print '-' * 20
        # -------------- region end : 3 & 4. 计算准确率和F1值 ---------------

        return y_pred,is_correct,accu,f1

    def print_model_descibe(self):
        import pprint
        detail = {'rand_seed': self.rand_seed,
                  'verbose': self.verbose,
                  'input_dim': self.input_dim,
                  'word_embedding_dim': self.word_embedding_dim,
                  'input_length': self.input_length,
                  'num_labels': self.num_labels,
                  'conv_filter_type': self.conv_filter_type,
                  'kmaxpooling_k': self.k,
                  'embedding_dropout_rate': self.embedding_dropout_rate,
                  'output_dropout_rate': self.output_dropout_rate,
                  'nb_epoch': self.nb_epoch,
                  'earlyStoping_patience': self.earlyStoping_patience,
                  }
        pprint.pprint(detail)
        logging.debug(detail)
        return detail

if __name__ == '__main__':
    # 使用样例
    train_X = ['你好', '无聊', '测试句子', '今天天气不错','我要买手机']
    trian_y = [1,3,2,2,3]
    test_X = ['句子','你好','你妹']
    test_y = [2,3,0]
    sentence_padding_length = 8
    feature_encoder = FeatureEncoder(train_data=train_X,
                                     sentence_padding_length=sentence_padding_length,
                                     verbose=0,
                                     need_segmented=True,
                                     full_mode=True,
                                     replace_number=True,
                                     remove_stopword=True,
                                     lowercase=True,
                                     padding_mode='center',
                                     add_unkown_word=True,
                                     mask_zero=True,
                                     )
    print feature_encoder.train_padding_index
    print map(feature_encoder.encoding_sentence,test_X)
    rand_embedding_cnn = RandEmbeddingCNN(
        rand_seed=1337,
        verbose=1,
        input_dim=feature_encoder.train_data_dict_size+1,
        word_embedding_dim=10,
        input_length = sentence_padding_length,
        num_labels = 5,
        conv_filter_type = [[100,2,10,'valid'],
                            [100,4,10,'valid'],
                            # [100,6,5,'valid'],
                            ],
        k=1,
        embedding_dropout_rate= 0.5,
        output_dropout_rate=0.5,
        nb_epoch=10,
        earlyStoping_patience = 5,
    )
    rand_embedding_cnn.print_model_descibe()
    # 训练模型
    rand_embedding_cnn.fit((feature_encoder.train_padding_index, trian_y),
                           (map(feature_encoder.encoding_sentence,test_X),test_y))
    rand_embedding_cnn.accuracy((map(feature_encoder.encoding_sentence,test_X),test_y))
    # 保存模型
    rand_embedding_cnn.save_model('model/modelA.pkl')

    quit()
    # 从保存的pickle中加载模型
    # rand_embedding_cnn.model_from_pickle('model/modelA.pkl')
    print rand_embedding_cnn.predict(feature_encoder.encoding_sentence('你好吗'))
