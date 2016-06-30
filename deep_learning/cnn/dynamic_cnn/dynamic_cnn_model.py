#encoding=utf8
__author__ = 'jdwang'
__date__ = 'create date: 2016-06-29'
__email__ = '383287471@qq.com'
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
import theano
from collections import OrderedDict
import lasagne
import DCNN

class DynamicCNN(object):
    '''
        DCNN模型: 两层CNN模型,每层都是一种类型卷积核，多核，随机初始化词向量,.
        使用 lasagne、theano实现。
        架构各个层次分别为: 输入层,embedding层,dropout层,卷积层,1-max pooling层,全连接层,dropout层,softmax层
        具体见:
            https://github.com/JDwangmo/coprocessor#2convolutional-neural-networks-for-sentence-classification
    '''

    def __init__(self,
                 verbose=0,
                 batch_size=4,
                 vocab_size=None,
                 word_embedding_dim=None,
                 conv_filter_type=None,
                 ktop = 1,
                 num_labels = None,
                 output_dropout_rate = 0.5,

                 rand_seed=1337,
                 input_length = None,
                 embedding_dropout_rate = 0.5,
                 nb_epoch=100,
                 earlyStoping_patience = 50,
                 ):
        '''
            1. 初始化参数
            2. 构建模型

        :param verbose: 数值越大,输出更详细的信息
        :type verbose: int
        :param batch_size: 一个批量batch的大小，默认为4
        :type batch_size: int
        :param vocab_size: 字典大小 ,即embedding层输入(onehot)的维度-1,embedding层输入的维度比字典大小多1的原因，是留出索引0给填充用。
        :type vocab_size: int
        :param word_embedding_dim: cnn设置选项,embedding层词向量的维度(长度).
        :type word_embedding_dim: int
        :param conv_filter_type: cnn设置选项,卷积层的类型.

            for example:每个列表代表某一层的卷积核类型(size)的卷积核,每层只有一种类型，但可以多核：
                conv_filter_type = [[100, 4, 'full'],
                                    [100, 5, 'full'],
                                   ]
        :type conv_filter_type: array-like
        :param ktop: cnn设置选项,dynamic k-max pooling层的的ktop(最后一层max-pooling层的size)值,默认为1
        :type ktop: int
        :param num_labels: cnn设置选项,最后输出层的大小,即分类类别的个数.
        :type num_labels: int
        :param output_dropout_rate: cnn设置选项,dropout层的的dropout rate,对输出层进入dropuout,如果为0,则不dropout
        :type output_dropout_rate: float

        :param rand_seed: 随机种子,假如设置为为None时,则随机取随机种子
        :type rand_seed: int
        :param input_length: cnn设置选项,输入句子(序列)的长度.
        :type input_length: int
        :param embedding_dropout_rate: cnn设置选项,dropout层的的dropout rate,对embedding层进入dropuout,如果为0,则不dropout
        :type embedding_dropout_rate: float
        :param nb_epoch: cnn设置选项,cnn迭代的次数.
        :type nb_epoch: int
        :param earlyStoping_patience: cnn设置选项,earlyStoping的设置,如果迭代次数超过这个耐心值,依旧不下降,则stop.
        :type earlyStoping_patience: int
        '''

        self.verbose = verbose
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.conv_filter_type = conv_filter_type
        self.ktop = ktop
        self.num_labels = num_labels
        self.output_dropout_rate = output_dropout_rate

        self.rand_seed = rand_seed
        self.verbose = verbose
        self.input_length = input_length
        self.embedding_dropout_rate = embedding_dropout_rate
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
            构建CNN模型,分别是：
                1. 输入层： (batch_size,None)
                2. embedding层
                3. 第一层卷积层： wide 1-dim convolution
                4. folding层：将相邻两维相加


        :return:
        '''
        # 1.输入层
        # 因为输入可以变长，所以 第2维为 None
        l_in = lasagne.layers.InputLayer(
            shape=(self.batch_size, None),
        )
        # 2. embedding层
        # 将 索引 投影成 向量，first表示 将0作为填充字符
        l_embedding = DCNN.embeddings.SentenceEmbeddingLayer(
            l_in,
            self.vocab_size,
            self.word_embedding_dim,
            padding='first'
        )
        # 3. 第一层卷积层： wide 1-dim convolution
        l_conv1 = DCNN.convolutions.Conv1DLayerSplitted(
            l_embedding,
            num_filters=self.conv_filter_type[0][0],
            filter_size =self.conv_filter_type[0][1],
            nonlinearity=lasagne.nonlinearities.linear,
            border_mode=self.conv_filter_type[0][2]
        )
        # 4. folding层：将相邻两维相加
        l_fold1 = DCNN.folding.FoldingLayer(l_conv1)
        # 5. 第一层max-pooling层
        l_pool1 = DCNN.pooling.DynamicKMaxPoolLayer(l_fold1,
                                                    ktop=self.ktop,
                                                    nroflayers=2,
                                                    layernr=1)

        l_nonlinear1 = lasagne.layers.NonlinearityLayer(l_pool1,
                                                        nonlinearity=lasagne.nonlinearities.rectify)
        # 第二层卷积层
        l_conv2 = DCNN.convolutions.Conv1DLayerSplitted(
            l_nonlinear1,
            num_filters=self.conv_filter_type[1][0],
            filter_size=self.conv_filter_type[1][1],
            nonlinearity=lasagne.nonlinearities.linear,
            border_mode=self.conv_filter_type[1][2]
        )


        l_fold2 = DCNN.folding.FoldingLayer(l_conv2)

        l_pool2 = DCNN.pooling.KMaxPoolLayer(l_fold2, self.ktop)

        l_nonlinear2 = lasagne.layers.NonlinearityLayer(l_pool2,
                                                        nonlinearity=lasagne.nonlinearities.rectify)

        l_dropout2 = lasagne.layers.DropoutLayer(l_nonlinear2, p=self.output_dropout_rate)

        output_layer = lasagne.layers.DenseLayer(
            l_dropout2,
            num_units=self.num_labels,
            nonlinearity=lasagne.nonlinearities.softmax
        )

        # allocate symbolic variables for the data
        X_batch = T.matrix('x',dtype='int64')
        y_batch = T.vector('y',dtype='int64')

        # Kalchbrenner uses a fine-grained L2 regularization in the Matlab code, default values taken from Matlab code
        # Training objective
        l2_layers = []
        for layer in lasagne.layers.get_all_layers(output_layer):
            if isinstance(layer, (DCNN.embeddings.SentenceEmbeddingLayer,
                                  DCNN.convolutions.Conv1DLayerSplitted,
                                  lasagne.layers.DenseLayer)):
                l2_layers.append(layer)

        # 计算训练误差:期望交叉商+L2正则
        loss_train = lasagne.objectives.aggregate(
            lasagne.objectives.categorical_crossentropy(lasagne.layers.get_output(output_layer, X_batch), y_batch),
            mode='mean')+ lasagne.regularization.regularize_layer_params_weighted(dict(zip(l2_layers, [0.0001,0.00003,0.000003,0.0001])),lasagne.regularization.l2)
        # validating/testing
        loss_eval = lasagne.objectives.categorical_crossentropy(
            lasagne.layers.get_output(output_layer, X_batch, deterministic=True), y_batch)
        cnn_pred = T.argmax(lasagne.layers.get_output(output_layer, X_batch, deterministic=True), axis=1)
        correct_predictions = T.eq(cnn_pred, y_batch)

        self.loss_trian = loss_train
        self.loss_eval = loss_eval
        self.cnn_pred = cnn_pred
        # In the matlab code, Kalchbrenner works with a adagrad reset mechanism, if the para --adagrad_reset has value 0, no reset will be applied
        all_params = lasagne.layers.get_all_params(output_layer)
        # updates, accumulated_grads = self.adagrad(loss_train, all_params, 0.001)
        updates = lasagne.updates.adagrad(loss_train, all_params, 1e-6)


        self.train_model = theano.function(inputs=[X_batch, y_batch], outputs=loss_train, updates=updates)

        self.valid_model = theano.function(inputs=[X_batch, y_batch], outputs=correct_predictions)

        self.test_model = theano.function(inputs=[X_batch, y_batch], outputs=correct_predictions)


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
        :param train_data: 训练数据,格式为:(train_X, train_y),train_X中每个句子以字典索引的形式表示,train_y是一个整形列表.
        :type train_data: (array-like,array-like)
        :param validation_data: 验证数据,格式为:(validation_X, validation_y),validation_X中每个句子以字典索引的形式表示,validation_y是一个整形列表.
        :type validation_data: (array-like,array-like)
        :return:
        '''
        batch_size = self.batch_size

        train_X, train_y = train_data
        train_X = np.asarray(train_X)
        validation_X, validation_y = validation_data
        validation_X = np.asarray(validation_X)

        n_train_samples = len(train_X)
        n_validation_samples = len(validation_X)

        print train_X[:5]
        print len(train_X)
        train_X = np.concatenate((train_X,
                                  np.asarray([np.zeros(len(train_X[0]),dtype=int)]*(batch_size - n_train_samples%batch_size))),
                                 axis=0,
                                 )
        train_y = np.concatenate((train_y,
                                  np.zeros(batch_size - n_train_samples%batch_size,dtype=int)),
                                 axis=0,
                                 )
        n_train_batches = len(train_X)/batch_size
        print n_train_batches

        validation_X = np.concatenate((validation_X,
                                  np.asarray([np.zeros(len(validation_X[0]), dtype=int)] * (
                                  batch_size - n_validation_samples % batch_size))),
                                 axis=0,
                                 )

        validation_y = np.concatenate((validation_y,
                              np.zeros(batch_size - n_validation_samples % batch_size, dtype=int)),
                             axis=0,
                             )
        n_dev_batches = len(validation_X)/batch_size


        best_validation_accuracy = 0
        epoch = 0
        while (epoch < self.nb_epoch):
            epoch = epoch + 1
            permutation = np.random.RandomState(self.rand_seed).permutation(n_train_batches)
            batch_counter = 0
            train_loss = 0

            for minibatch_index in permutation:


                x_input = train_X[minibatch_index * batch_size:(minibatch_index + 1) * batch_size]
                y_input = train_y[minibatch_index * batch_size:(minibatch_index + 1) * batch_size]
                train_loss += self.train_model(x_input, y_input)

                # print x_input
                # print y_input
                print batch_counter
                print train_loss
                # quit()
                if batch_counter > 0 :
                    accuracy_valid = []
                    for minibatch_dev_index in range(n_dev_batches):
                        x_input = validation_X[
                                  minibatch_dev_index * batch_size:(minibatch_dev_index + 1) * batch_size]
                        y_input = validation_y[
                                  minibatch_dev_index * batch_size:(minibatch_dev_index + 1) * batch_size]
                        accuracy_valid.append(self.valid_model(x_input, y_input))

                    # dirty code to correctly asses validation accuracy, last results in the array are predictions for the padding rows and can be dumped afterwards
                    this_validation_accuracy = np.concatenate(accuracy_valid)[0:n_validation_samples].sum() / float(n_validation_samples)

                    if this_validation_accuracy > best_validation_accuracy:
                        print(
                        "Train loss, " + str((train_loss / 1)) + ", validation accuracy: " + str(
                            this_validation_accuracy * 100) + "%")
                        best_validation_accuracy = this_validation_accuracy
                    #
                    #     # test it
                    #     accuracy_test = []
                    #     for minibatch_test_index in range(n_test_batches):
                    #         x_input = test_x_indexes_extended[
                    #                   minibatch_test_index * batch_size:(minibatch_test_index + 1) * batch_size,
                    #                   0:test_lengths[(minibatch_test_index + 1) * batch_size - 1]]
                    #         y_input = test_y_extended[
                    #                   minibatch_test_index * batch_size:(minibatch_test_index + 1) * batch_size]
                    #         accuracy_test.append(test_model(x_input, y_input))
                    #     this_test_accuracy = numpy.concatenate(accuracy_test)[0:n_test_samples].sum() / float(
                    #         n_test_samples)
                    #     print("Test accuracy: " + str(this_test_accuracy * 100) + "%")

                train_loss = 0
                batch_counter += 1

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



        y_pred = self.model_output([test_X,0])[0]
        y_pred = y_pred.argmax(axis=-1)

        is_correct = y_pred==test_y
        logging.debug('正确的个数:%d'%(sum(is_correct)))
        print '正确的个数:%d'%(sum(is_correct))
        accu = sum(is_correct)/(1.0*len(test_y))
        logging.debug('准确率为:%f'%(accu))
        print '准确率为:%f'%(accu)

        return y_pred,is_correct,accu

    def print_model_descibe(self):
        import pprint
        detail = {'rand_seed': self.rand_seed,
                  'verbose': self.verbose,
                  'input_dim': self.vocab_size,
                  'word_embedding_dim': self.word_embedding_dim,
                  'input_length': self.input_length,
                  'num_labels': self.num_labels,
                  'conv_filter_type': self.conv_filter_type,
                  'kmaxpooling_ktop': self.ktop,
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
    test_y = [3,1,1]
    sentence_padding_length = 8
    feature_encoder = FeatureEncoder(train_data=train_X,
                                     sentence_padding_length=sentence_padding_length,
                                     verbose=0)
    print feature_encoder.train_padding_index
    print map(feature_encoder.encoding_sentence,test_X)
    rand_embedding_cnn = DynamicCNN(
        rand_seed=1337,
        verbose=1,
        batch_size=2,
        vocab_size=feature_encoder.train_data_dict_size,
        word_embedding_dim=10,
        input_length = sentence_padding_length,
        num_labels = 5,
        conv_filter_type = [[100,2,'full'],
                            [100,4,'full'],
                            # [100,6,5,'valid'],
                            ],
        ktop=2,
        embedding_dropout_rate= 0.5,
        output_dropout_rate=0.5,
        nb_epoch=10,
        earlyStoping_patience = 5,
    )
    rand_embedding_cnn.print_model_descibe()
    # 训练模型
    rand_embedding_cnn.fit((feature_encoder.train_padding_index, trian_y),
                           (map(feature_encoder.encoding_sentence,test_X),test_y))
    quit()
    # 保存模型
    rand_embedding_cnn.save_model('model/modelA.pkl')

    quit()
    # 从保存的pickle中加载模型
    # rand_embedding_cnn.model_from_pickle('model/modelA.pkl')
    # print rand_embedding_cnn.predict(feature_encoder.encoding_sentence('你好吗'))