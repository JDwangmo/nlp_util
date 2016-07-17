# encoding=utf8
from __future__ import print_function

import time
from sklearn.metrics import f1_score

__author__ = 'jdwang'
__date__ = 'create date: 2016-06-29'
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
from base.common_model_class import CommonModel
# theano.config.compute_test_value = 'warn'
# theano.config.optimizer = 'fast_compile'
# theano.config.exception_verbosity='high'
class DynamicCNN(CommonModel):
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
                 nb_epoch=100,

                 input_length=None,
                 embedding_dropout_rate=0.5,
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
        :param nb_epoch: cnn设置选项,cnn迭代的次数.
        :type nb_epoch: int


        :param embedding_dropout_rate: cnn设置选项,dropout层的的dropout rate,对embedding层进入dropuout,如果为0,则不dropout
        :type embedding_dropout_rate: float
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
        self.nb_epoch = nb_epoch

        self.rand_seed = rand_seed
        self.verbose = verbose
        self.input_length = input_length
        self.embedding_dropout_rate = embedding_dropout_rate
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

    def batch_predict_bestn(self, sentences, bestn=1):
        '''
            批量预测句子的类别,对输入的句子进行预测

        :param sentences: 测试句子,
        :type sentences: array-like
        :param transform: 是否转换句子，如果为True,输入原始字符串句子即可，内部已实现转换。
        :type transform: bool
        :param bestn: 预测，并取出bestn个结果。
        :type bestn: int
        '''
        # if transform_input:
        #     sentences = self.transform(sentences)
        sentences = np.asarray(sentences)
        # print(sentences)
        assert len(sentences.shape) == 2, 'shape必须是2维的！'

        y_pred_prob = self.batch_predict_proba(sentences)
        y_pred_result = y_pred_prob.argsort(axis=-1)[:, ::-1][:, :bestn]
        y_pred_score = np.asarray([score[index] for score, index in zip(y_pred_prob, y_pred_result)])
        return y_pred_result, y_pred_score

    def batch_predict_proba(self, sentences, bestn=1):
        '''
            批量预测,并返回每个类的概率：输入多个句子索引，一起预测结果，主要步骤如下：
                1. 检查数据合法性和转换格式等;
                2. 对数据进行补全，直到是 self.batch_size的倍数;
                3. 预测，并取出相应的类别

        :param sentences: 测试句子,
        :type sentences: array-like
        :param bestn: 预测，并取出各个类别的概率。
        :type bestn: int
        :return:
        '''
        # -------------- region start : 1. 检查数据合法性和转换格式等; -------------
        if self.verbose > 2:
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('1. 检查数据合法性和转换格式等;')
            print('1. 检查数据合法性和转换格式等;')
        # -------------- code start : 开始 -------------

        sentences_index = np.asarray(sentences, dtype=np.int32)

        assert len(sentences_index.shape) == 2, '输入的维度一定要是2维！'

        # -------------- code start : 结束 -------------
        if self.verbose > 2:
            logging.debug('-' * 20)
            print('-' * 20)
        # -------------- region end : 1. 检查数据合法性和转换格式等; ---------------

        # -------------- region start : 2. 对数据进行补全，直到是 self.batch_size的倍数 -------------
        if self.verbose > 2:
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('2. 对数据进行补全，直到是 self.batch_size的倍数;')
            print('2. 对数据进行补全，直到是 self.batch_size的倍数; ')
        # -------------- code start : 开始 -------------

        # 样例个数
        num_sentences, sentence_length = sentences_index.shape
        # 需要补全多个样例
        num_need_padding = self.batch_size - num_sentences % self.batch_size

        batch_x_input = np.concatenate(
            (sentences_index, np.zeros((num_need_padding, sentence_length), dtype=np.int32)), axis=0)

        batch_y_input = np.zeros(len(batch_x_input))

        # -------------- code start : 结束 -------------
        if self.verbose > 2:
            logging.debug('-' * 20)
            print('-' * 20)
        # -------------- region end : 2. 对数据进行补全，直到是 self.batch_size的倍数; ---------------

        # -------------- region start : 3.预测 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('3.预测')
            print('3.预测')
        # -------------- code start : 开始 -------------
        batch_y_predict = []
        for batch in self.iterate_minibatches(batch_x_input, batch_y_input, self.batch_size, shuffle=False):
            inputs, targets = batch
            y_predict_prob = self.test_prediction_fn(inputs)
            # print(y_predict)
            batch_y_predict.extend(y_predict_prob)

        batch_y_predict = batch_y_predict[:num_sentences]
        # print(batch_y_predict)

        # -------------- code start : 结束 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
        # -------------- region end : 3.预测 ---------------
        return np.asarray(batch_y_predict)

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    def build_dcnn_network(self,input_var):
        '''
            构建 DCNN 网络,分别是：
                1. 输入层：
                    - 输出shape： (batch_size,input_length)，比如:(32,8)

                2. embedding层：
                    - 输入shape： (batch_size,input_length)，比如:(32,8)
                    - 输出shape：(batch_size,input_dim,input_length)，比如:(32,48,8)

                3. 第一层卷积层： wide 1-dim convolution
                    - 输入shape：(batch_size,input_dim,input_length)，比如:(32,48,8)
                    - filter shape： (filter_num,filter_size)，比如:(100,3)
                    - 输出shape：(batch_size,filter_num,input_dim,（input_length-filter_size+1）)，比如:(32,100,48,6)

                4. folding层：将相邻两维相加
                    - 输入shape： (batch_size,filter_num,input_dim,input_length)，比如:(32,100,48,6)
                    - 输出shape：(batch_size,filter_num,input_dim/2,input_length)，比如:(32,100,24,6)

                5. 第一层 max-pooling 层： Dynamic K max pooling
                    - 输入shape：(batch_size,filter_num,input_dim,input_length)，比如:(32,100,24,6)
                    - 输出shape：
                        - 计算 k ：ceil(input_length*（1/2）
                        - (batch_size,filter_num,input_dim,k)，比如:(32,100,24,3)
                6. ReLu激活函数：
                7. 第二层卷积层：wide 1-dim convolution
                8. 第二层fold层：
                9. 第二层 max pooling 层（K-max）
                10. ReLu激活函数：
                11. Dropout layers
                12. Softmax 输出层



        :return:
        '''

        # 1.输入层
        # 因为输入可以变长，所以 第2维为 None
        l1_in = lasagne.layers.InputLayer(
            shape=(self.batch_size, None),
            input_var=input_var,
        )
        # 2. embedding层
        # 将 索引 投影成 向量，first表示 将0作为填充字符
        l2_embedding = DCNN.embeddings.SentenceEmbeddingLayer(
            l1_in,
            self.vocab_size,
            self.word_embedding_dim,
            padding='first'
        )
        # 3. 第一层卷积层： wide 1-dim convolution
        l3_conv = DCNN.convolutions.Conv1DLayerSplitted(
            l2_embedding,
            num_filters=self.conv_filter_type[0][0],
            filter_size=self.conv_filter_type[0][1],
            nonlinearity=lasagne.nonlinearities.linear,
            border_mode=self.conv_filter_type[0][2]
        )
        # 4. folding层：将相邻两维相加
        l4_fold = DCNN.folding.FoldingLayer(l3_conv)
        # 5. 第一层max-pooling层
        l5_pool = DCNN.pooling.DynamicKMaxPoolLayer(l4_fold,
                                                    ktop=self.ktop,
                                                    nroflayers=2,
                                                    layernr=1)
        # 6. ReLu激活函数
        l6_nonlinear = lasagne.layers.NonlinearityLayer(l5_pool,
                                                        nonlinearity=lasagne.nonlinearities.rectify)
        # 7. 第二层卷积层
        l7_conv = DCNN.convolutions.Conv1DLayerSplitted(
            l6_nonlinear,
            num_filters=self.conv_filter_type[1][0],
            filter_size=self.conv_filter_type[1][1],
            nonlinearity=lasagne.nonlinearities.linear,
            border_mode=self.conv_filter_type[1][2]
        )
        # 8. 第二层fold层
        l8_fold = DCNN.folding.FoldingLayer(l7_conv)

        # 9. 第二层 max pooling 层（K-max）
        l9_pool = DCNN.pooling.KMaxPoolLayer(l8_fold, self.ktop)
        # 10. ReLu激活函数
        l10_nonlinear = lasagne.layers.NonlinearityLayer(l9_pool,
                                                        nonlinearity=lasagne.nonlinearities.rectify)
        # 11. Dropout layers

        l11_dropout = lasagne.layers.DropoutLayer(l10_nonlinear, p=self.output_dropout_rate)

        # 12. Softmax 输出层
        l12_output_layer = lasagne.layers.DenseLayer(
            l11_dropout,
            num_units=self.num_labels,
            nonlinearity=lasagne.nonlinearities.softmax
        )


        return l12_output_layer

    def build_model(self):
        '''
            构建 DCNN模型，包括 目标函数 训练方法等

        :return:
        '''

        # 设置模型的输入和输出的Theano变量
        input_var = T.matrix('inputs', dtype='int32')
        target_var = T.vector('targets', dtype='int32')
        # 获取模型网络（结构图）
        network = self.build_dcnn_network(input_var)

        # 模型的预测结果
        prediction = lasagne.layers.get_output(network)
        # 目标损失函数：期望交叉商
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()
        # We could add some weight decay as well here, see lasagne.regularization.

        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
        # 获取模型的参数，并设置更新方法（SGD）
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

        # Create a loss expression for validation/testing.
        # disabling dropout layers.
        # 获取预测结果，设置 deterministic为True 关闭 drpout 层
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
        test_loss = test_loss.mean()
        # As a bonus, also create an expression for the classification accuracy:
        # 准确率
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                          dtype=theano.config.floatX)

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_fn = theano.function([input_var, target_var], loss, updates=updates)

        # Compile a second function computing the validation loss and accuracy:
        val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

        test_prediction_fn = theano.function([input_var],test_prediction)

        self.test_prediction_fn = test_prediction_fn
        self.train_fn = train_fn
        self.val_fn = val_fn

    def fit(self, train_data=None, validation_data=None):
        # type: (array, array) -> object
        '''
            cnn model 的训练
                1. 对数据进行格式转换
                2. 模型训练
        :param train_data: 训练数据,格式为:(train_X, train_y),train_X中每个句子以字典索引的形式表示,train_y是一个整形列表.
        :type train_data: (array-like,array-like)
        :param validation_data: 验证数据,格式为:(validation_X, validation_y),validation_X中每个句子以字典索引的形式表示,validation_y是一个整形列表.
        :type validation_data: (array-like,array-like)
        :return:
        '''

        # -------------- region start : 1. 对数据进行格式转换 -------------
        if self.verbose > 2 :
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('1. 对数据进行格式转换')
            print('1. 对数据进行格式转换')
        # -------------- code start : 开始 -------------

        train_X, train_y = train_data
        train_X = np.asarray(train_X, dtype=np.int32)
        train_y = np.asarray(train_y, dtype=np.int32)

        validation_X, validation_y = validation_data
        validation_X = np.asarray(validation_X, dtype=np.int32)
        validation_y = np.asarray(validation_y, dtype=np.int32)

        # -------------- code start : 结束 -------------
        if self.verbose > 2 :
            logging.debug('-' * 20)
            print('-' * 20)
        # -------------- region end : 1. 对数据进行格式转换 ---------------
        # -------------- region start : 2. 模型训练 -------------
        if self.verbose > 2 :
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('2. 模型训练')
            print('2. 模型训练')
        # -------------- code start : 开始 -------------

        print("Starting training...")
        # We iterate over epochs:
        for epoch in range(self.nb_epoch):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in self.iterate_minibatches(train_X, train_y, self.batch_size, shuffle=True):
                inputs, targets = batch
                train_err += self.train_fn(inputs, targets)
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in self.iterate_minibatches(validation_X, validation_y, self.batch_size, shuffle=False):
                inputs, targets = batch
                err, acc = self.val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, self.nb_epoch, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))

        # After training, we compute and print the test error:
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in self.iterate_minibatches(validation_X, validation_y, self.batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = self.val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
        print("Final results:")
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100))

        # Optionally, you could now dump the network weights to a file like this:
        # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
        #
        # And load them again later on like this:
        # with np.load('model.npz') as f:
        #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        # lasagne.layers.set_all_param_values(network, param_values)


        # -------------- code start : 结束 -------------
        if self.verbose > 2:
            logging.debug('-' * 20)
            print('-' * 20)
        # -------------- region end : 2. 模型训练 ---------------

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

    def batch_predict(self, sentences):
        '''
            批量预测句子的类别,对输入的句子进行预测

        :param sentences: 测试句子,
        :type sentences: array-like
        :param transform: 是否转换句子，如果为True,输入原始字符串句子即可，内部已实现转换。
        :type transform: bool
        '''
        y_pred,_ = self.batch_predict_bestn(sentences,1)
        y_pred = np.asarray(y_pred).flatten()

        return y_pred



    def predict(self, sentence):
        '''
            预测,对输入的句子进行预测

        :param sentence_index: 测试句子,以字典索引的形式
        :type sentence_index: array-like
        '''

        y_pred = self.batch_predict([sentence])[0]
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

        y_pred = self.batch_predict(sentences=test_X)

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
    feature_encoder = FeatureEncoder(
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
    train_X_features = feature_encoder.fit_transform(train_data=train_X)
    test_X_features = feature_encoder.transform(test_X)
    print(train_X_features)
    dcnn = DynamicCNN(
        rand_seed=1337,
        verbose=1,
        batch_size=2,
        vocab_size=feature_encoder.vocabulary_size,
        word_embedding_dim=48,
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
    dcnn.fit((train_X_features, trian_y),
             (test_X_features, test_y))
    print(dcnn.predict(feature_encoder.transform_sentence(test_X[0])))
    dcnn.accuracy((test_X_features, test_y))
    print(dcnn.batch_predict(test_X_features))
    # 保存模型
    # dcnn.save_model('model/modelA.pkl')

    # 从保存的pickle中加载模型
    # print onehot_cnn.predict(feature_encoder.transform_sentence('你好吗'))
