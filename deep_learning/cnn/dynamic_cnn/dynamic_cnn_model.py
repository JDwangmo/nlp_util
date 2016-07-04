# encoding=utf8
from __future__ import print_function

from sklearn.metrics import f1_score

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
# theano.config.compute_test_value = 'warn'
# theano.config.optimizer = 'fast_compile'
# theano.config.exception_verbosity='high'
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
                 rand_seed=1337,
                 batch_size=4,
                 vocab_size=None,
                 word_embedding_dim=None,
                 conv_filter_type=None,
                 ktop=1,
                 num_labels=None,
                 output_dropout_rate=0.5,

                 input_length=None,
                 embedding_dropout_rate=0.5,
                 nb_epoch=100,
                 earlyStoping_patience=50,
                 ):
        '''
            1. 初始化参数
            2. 构建模型

        :param verbose: 数值越大,输出更详细的信息
        :type verbose: int
        :param rand_seed: 随机种子,for Reproducibility. 假如设置为为None时,则随机取随机种子
        :type rand_seed: int
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
        :param input_length: cnn设置选项,输入句子(序列)的长度.若为None的话，则句子输入可以不等长。
        :type input_length: int / None


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
        self.earlyStoping_patience = earlyStoping_patience

        # cnn model 的预测模型
        self.prediction = None
        # cnn model 的训练模型
        self.train_model = None
        # cnn model 的验证模型
        self.valid_model = None
        # cnn model 的测试模型
        self.test_model = None
        if self.rand_seed is not None:
            np.random.seed(self.rand_seed)
        # 构建模型
        self.build_model()


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
            shape=(self.batch_size, self.input_length),
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
            filter_size=self.conv_filter_type[0][1],
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
        X_batch = T.matrix('x', dtype='int32')
        y_batch = T.vector('y', dtype='int32')

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
            mode='mean') + lasagne.regularization.regularize_layer_params_weighted(
            dict(zip(l2_layers, [0.0001, 0.00003, 0.000003, 0.0001])), lasagne.regularization.l2)
        # validating/testing
        loss_eval = lasagne.objectives.categorical_crossentropy(
            lasagne.layers.get_output(output_layer, X_batch, deterministic=True), y_batch)
        cnn_pred = T.argmax(lasagne.layers.get_output(output_layer, X_batch, deterministic=True), axis=1)
        correct_predictions = T.eq(cnn_pred, y_batch)
        predictions_accu = T.sum(correct_predictions) / (y_batch.shape[0] * 1.0)

        self.loss_trian = loss_train
        self.loss_eval = loss_eval
        self.cnn_pred = cnn_pred
        # In the matlab code, Kalchbrenner works with a adagrad reset mechanism, if the para --adagrad_reset has value 0, no reset will be applied
        all_params = lasagne.layers.get_all_params(output_layer)
        # updates, accumulated_grads = self.adagrad(loss_train, all_params, 0.001)
        updates = lasagne.updates.adagrad(loss_train, all_params, 1e-6)

        self.prediction = theano.function([X_batch], outputs=cnn_pred)

        self.train_model = theano.function(inputs=[X_batch, y_batch], outputs=[loss_train, predictions_accu],
                                           updates=updates)

        self.valid_model = theano.function(inputs=[X_batch, y_batch], outputs=correct_predictions)

        self.test_model = theano.function(inputs=[X_batch, y_batch], outputs=correct_predictions)


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
        train_X = np.asarray(train_X,dtype=np.int32)
        train_y = np.asarray(train_y,dtype=np.int32)
        validation_X, validation_y = validation_data
        validation_X = np.asarray(validation_X,dtype=np.int32)
        validation_y = np.asarray(validation_y,dtype=np.int32)

        n_train_samples = len(train_X)
        n_validation_samples = len(validation_X)

        # print train_X[:5]
        # print len(train_X)
        # train_X = np.concatenate((train_X,
        #                           np.asarray([np.zeros(len(train_X[0]), dtype=int)] * (
        #                           batch_size - n_train_samples % batch_size))),
        #                          axis=0,
        #                          )
        # train_y = np.concatenate((train_y,
        #                           np.zeros(batch_size - n_train_samples % batch_size, dtype=int)),
        #                          axis=0,
        #                          )
        n_train_batches = len(train_X) / batch_size
        #
        # validation_X = np.concatenate((validation_X,
        #                                np.asarray([np.zeros(len(validation_X[0]), dtype=int)] * (
        #                                    batch_size - n_validation_samples % batch_size))),
        #                               axis=0,
        #                               )
        #
        # validation_y = np.concatenate((validation_y,
        #                                np.zeros(batch_size - n_validation_samples % batch_size, dtype=int)),
        #                               axis=0,
        #                               )
        n_validation_batches = len(validation_X) / batch_size

        best_validation_accuracy = 0
        epoch = 0
        while (epoch < self.nb_epoch):
            epoch = epoch + 1
            permutation = np.random.permutation(n_train_batches)
            # batch_counter = 0
            train_loss = 0
            # print batch_counter

            for counter,minibatch_index in enumerate(permutation):
                x_input = train_X[minibatch_index * batch_size:(minibatch_index + 1) * batch_size]
                y_input = train_y[minibatch_index * batch_size:(minibatch_index + 1) * batch_size]
                loss, train_accu = self.train_model(x_input, y_input)
                # print train_accu
                train_loss += loss
                # -------------- print start : just print info -------------
                if self.verbose > 2 :
                    logging.debug('已处理%d个数据。训练误差为：%f'%(counter*self.batch_size,train_loss))
                    print('已处理%d个数据。训练误差为：%f'%(counter*self.batch_size,train_loss))
                # -------------- print end : just print info -------------
                # batch_counter += 1

                # quit()
            # 验证
            accuracy_valid = []
            for minibatch_dev_index in range(n_validation_batches):
                x_input = validation_X[
                          minibatch_dev_index * batch_size:(minibatch_dev_index + 1) * batch_size]
                y_input = validation_y[
                          minibatch_dev_index * batch_size:(minibatch_dev_index + 1) * batch_size]
                accuracy_valid.append(self.valid_model(x_input, y_input))

            # dirty code to correctly asses validation accuracy, last results in the array are predictions for the padding rows and can be dumped afterwards
            this_validation_accuracy = np.concatenate(accuracy_valid)[0:n_validation_samples].sum() / float(
                n_validation_samples)
            # -------------- print start : just print info -------------

            if self.verbose > 1:
                logging.debug(
                    'epoch %d，' % (epoch) + "Train loss, " + str(train_loss) + ", validation accuracy: " + str(
                        this_validation_accuracy * 100) + "%")
                print('epoch %d，' % (epoch) + "Train loss, " + str(train_loss) + ", validation accuracy: " + str(this_validation_accuracy * 100) + "%")

                # -------------- print end : just print info -------------

    def save_model(self, path):
        '''
            保存模型,保存成pickle形式
        :param path: 模型保存的路径
        :type path: 模型保存的路径
        :return:
        '''
        pickle.dump(self.prediction, open(path, 'wb'))

    def model_from_pickle(self, path):
        '''
            从模型文件中直接加载模型
        :param path:
        :return: RandEmbeddingCNN object
        '''
        self.prediction = pickle.load(file(path, 'rb'))

    def batch_predict(self, sentences_index):
        '''
            批量预测：输入多个句子索引，一起预测结果，主要步骤如下啊：
                1. 检查数据合法性和转换格式等;
                2. 对数据进行补全，直到是 self.batch_size的倍数;
                3. 预测，并取出相应的类别

        :param sentences_index:
        :return:
        '''
        # -------------- region start : 1. 检查数据合法性和转换格式等; -------------
        if self.verbose > 2:
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('1. 检查数据合法性和转换格式等;')
            print('1. 检查数据合法性和转换格式等;')
        # -------------- code start : 开始 -------------

        sentences_index = np.asarray(sentences_index, dtype=np.int32)

        assert len(sentences_index.shape) == 2, '输入的维度一定要是2维！'


        # -------------- code start : 结束 -------------
        if self.verbose > 2:
            logging.debug('-' * 20)
            print('-' * 20)
        # -------------- region end : 1. 检查数据合法性和转换格式等; ---------------

        # -------------- region start : 2. 对数据进行补全，直到是 self.batch_size的倍数 -------------
        if self.verbose > 2 :
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('2. 对数据进行补全，直到是 self.batch_size的倍数;')
            print('2. 对数据进行补全，直到是 self.batch_size的倍数; ')
        # -------------- code start : 开始 -------------

        # 样例个数
        num_sentences, sentence_length = sentences_index.shape
        # 需要补全多个样例
        num_need_padding = self.batch_size - num_sentences % self.batch_size

        x_input = np.concatenate(
            (sentences_index, np.zeros((num_need_padding, sentence_length), dtype=np.int32)), axis=0)


        # -------------- code start : 结束 -------------
        if self.verbose > 2 :
            logging.debug('-' * 20)
            print('-' * 20)
        # -------------- region end : 2. 对数据进行补全，直到是 self.batch_size的倍数; ---------------

        # -------------- region start : 3.预测 -------------
        if self.verbose > 1 :
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('3.预测')
            print('3.预测')
        # -------------- code start : 开始 -------------

        n_batches = len(x_input) / self.batch_size
        y_pred = []
        for minibatch_index in range(n_batches):
            x_minibatch_input = x_input[minibatch_index * self.batch_size:(minibatch_index + 1) * self.batch_size]
            y_pred.extend(self.prediction(x_minibatch_input))

        y_pred = y_pred[:num_sentences]

        # -------------- code start : 结束 -------------
        if self.verbose > 1 :
            logging.debug('-' * 20)
            print('-' * 20)
        # -------------- region end : 3.预测 ---------------
        return np.asarray(y_pred)



    def predict(self, sentence_index):
        '''
            预测,对输入的句子进行预测

        :param sentence_index: 测试句子,以字典索引的形式
        :type sentence_index: array-like
        '''

        y_pred = self.batch_predict([sentence_index])[0]
        return y_pred

    def accuracy(self, test_data):
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
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('1. 转换格式')
            print('1. 转换格式')
        # -------------- code start : 开始 -------------

        test_X, test_y = test_data
        test_X = np.asarray(test_X)

        # -------------- code start : 结束 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
        # -------------- region end : 1. 转换格式 ---------------

        # -------------- region start : 2. 批量预测 -------------
        if self.verbose > 1 :
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('2. 批量预测')
            print('2. 批量预测')
        # -------------- code start : 开始 -------------

        y_pred = self.batch_predict(sentences_index=test_X)

        is_correct = y_pred == test_y

        # -------------- code start : 结束 -------------
        if self.verbose > 1 :
            logging.debug('-' * 20)
            print('-' * 20)
        # -------------- region end : 2. 批量预测 ---------------


        # -------------- region start : 3 & 4. 计算准确率和F1值 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('3 & 4. 计算准确率和F1值')
            print('3 & 4. 计算准确率和F1值')
        # -------------- code start : 开始 -------------

        is_correct = y_pred == test_y
        logging.debug('正确的个数:%d' % (sum(is_correct)))
        print('正确的个数:%d' % (sum(is_correct)))
        accu = sum(is_correct) / (1.0 * len(test_y))
        logging.debug('准确率为:%f' % (accu))
        print('准确率为:%f' % (accu))

        f1 = f1_score(test_y, y_pred.tolist(), average=None)
        logging.debug('F1为：%s' % (str(f1)))
        print('F1为：%s' % (str(f1)))

        # -------------- code start : 结束 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
        # -------------- region end : 3 & 4. 计算准确率和F1值 ---------------

        return y_pred, is_correct, accu, f1


        return y_pred, is_correct, accu

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
    train_X = ['你好', '无聊', '测试句子', '今天天气不错', '我要买手机']
    trian_y = [1, 3, 2, 2, 3]
    test_X = ['你好', '你好', '你妹']
    test_y = [3, 1, 1]
    sentence_padding_length = 10
    feature_encoder = FeatureEncoder(train_data=train_X,
                                     sentence_padding_length=sentence_padding_length,
                                     verbose=0,
                                     need_segmented=True,
                                     full_mode=True,
                                     remove_stopword=True,
                                     replace_number=True,
                                     lowercase=True,
                                     zhs2zht=True,
                                     remove_url=True,
                                     padding_mode='center',
                                     add_unkown_word=True,
                                     mask_zero=True
                                     )
    print(feature_encoder.train_padding_index)
    # print map(feature_encoder.encoding_sentence,test_X)
    dcnn = DynamicCNN(
        rand_seed=1337,
        verbose=2,
        batch_size=1,
        vocab_size=feature_encoder.train_data_dict_size,
        word_embedding_dim=10,
        # input_length=None,
        input_length=sentence_padding_length,
        num_labels=4,
        conv_filter_type=[[100, 2, 'full'],
                          [100, 4, 'full'],
                          # [100,6,5,'valid'],
                          ],
        ktop=1,
        embedding_dropout_rate=0.5,
        output_dropout_rate=0.5,
        nb_epoch=10,
        earlyStoping_patience=5,
    )
    dcnn.print_model_descibe()
    # 训练模型
    # dcnn.model_from_pickle('model/modelA.pkl')
    dcnn.fit((feature_encoder.train_padding_index, trian_y),
             (map(feature_encoder.encoding_sentence, test_X), test_y))
    print(dcnn.predict(feature_encoder.encoding_sentence(test_X[0])))
    dcnn.accuracy((map(feature_encoder.encoding_sentence, test_X), test_y))
    print(dcnn.batch_predict(map(feature_encoder.encoding_sentence, test_X)))
    # 保存模型
    # dcnn.save_model('model/modelA.pkl')

    # 从保存的pickle中加载模型
    # print rand_embedding_cnn.predict(feature_encoder.encoding_sentence('你好吗'))
