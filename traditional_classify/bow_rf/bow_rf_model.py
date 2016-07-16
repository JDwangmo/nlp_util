#encoding=utf8
from __future__ import print_function


__author__ = 'jdwang'
__date__ = 'create date: 2016-07-06'
__email__ = '383287471@qq.com'
import numpy as np
import pandas as pd
import logging
import timeit
from sklearn.ensemble import RandomForestClassifier
from base.common_model_class import CommonModel
from data_processing_util.feature_encoder.bow_feature_encoder import FeatureEncoder
from sklearn.metrics import f1_score
import cPickle as pickle

class BowRandomForest(CommonModel):
    '''
        使用BOW或者TFIDF作为特征输入，使用random forest作为分类器
    '''

    def __init__(self,
                 rand_seed=1337,
                 verbose=0,
                 n_estimators = 200,
                 min_samples_leaf = 2,
                 feature_encoder=None,

                 ):
        '''
            1. 初始化参数，并检验参数合法性。
            2. 设置随机种子，构建模型

        :param rand_seed: 随机种子,假如设置为为None时,则随机取随机种子
        :type rand_seed: int
        :param verbose: 数值越大,输出更详细的信息
        :type verbose: int
        :param rand_seed: 随机种子,假如设置为为None时,则随机取随机种子
        :type rand_seed: int
        :param verbose: 数值越大,输出更详细的信息
        :type verbose: int
        :param feature_encoder: 输入数据的设置选项，设置输入编码器
        :type feature_encoder: bow_feature_encoder.FeatureEncoder
        '''

        self.rand_seed = rand_seed
        self.verbose = verbose
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.feature_encoder = feature_encoder

        # random forest model
        self.model = None

    def transform(self, data):
        '''
            批量转换数据转换数据

        :param train_data: array-like,1D
        :return:features
        '''

        features = self.feature_encoder.transform(data)
        return features

    def model_from_pickle(self, path):
        '''
            从模型文件中直接加载模型
        :param path:
        :return: RandEmbeddingCNN object
        '''
        fin = file(path, 'rb')
        self.model = pickle.load(fin)
        self.feature_encoder = pickle.load(fin)
        return self

    def save_model(self, path):
        '''
            保存模型,保存成pickle形式
        :param path: 模型保存的路径
        :type path: 模型保存的路径
        :return:
        '''
        fout = open(path, 'wb')
        pickle.dump(self.model,fout)
        pickle.dump(self.feature_encoder, fout)


    def batch_predict_bestn(self, sentences,transform_input=False, bestn=1):
        '''
            批量预测句子的类别,对输入的句子进行预测

        :param sentences: 测试句子,
        :type sentences: array-like
        :param transform: 是否转换句子，如果为True,输入原始字符串句子即可，内部已实现转换。
        :type transform: bool
        :param bestn: 预测，并取出bestn个结果。
        :type bestn: int
        '''
        if transform_input:
            sentences = self.transform(sentences)
        sentences = np.asarray(sentences)
        # print(sentences)
        assert len(sentences.shape) == 2, 'shape必须是2维的！'

        y_pred_prob = self.model.predict_proba(sentences)
        y_pred_result = y_pred_prob.argsort(axis=-1)[:,::-1][:,:bestn]
        y_pred_score = np.asarray([score[index] for score,index in zip(y_pred_prob,y_pred_result)])
        return y_pred_result,y_pred_score

    def batch_predict(self, sentences,transform_input=False):
        '''
            批量预测句子的类别,对输入的句子进行预测

        :param sentences: 测试句子,
        :type sentences: array-like
        :param transform: 是否转换句子，如果为True,输入原始字符串句子即可，内部已实现转换。
        :type transform: bool
        '''
        y_pred,_ = self.batch_predict_bestn(sentences,transform_input,1)
        y_pred = np.asarray(y_pred).flatten()

        return y_pred

    def fit(self, train_data=None, validation_data=None):
        '''
            rf model 的训练
                1. 对数据进行格式转换,
                2. 模型训练

        :param train_data: 训练数据,格式为:(train_X, train_y),train_X中每个句子以字典索引的形式表示(使用data_processing_util.feature_encoder.onehot_feature_encoder编码器编码),train_y是一个整形列表.
        :type train_data: (array-like,array-like)
        :param validation_data: 验证数据,格式为:(validation_X, validation_y),validation_X中每个句子以字典索引的形式表示(使用data_processing_util.feature_encoder.onehot_feature_encoder编码器编码),validation_y是一个整形列表.
        :type validation_data: (array-like,array-like)
        :return: None
        '''
        # -------------- region start : 1. 对数据进行格式转换 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('1. 对数据进行格式转换')
            print('1. 对数据进行格式转换')
        # -------------- code start : 开始 -------------


        train_X, train_y = train_data
        train_X = np.asarray(train_X)
        validation_X, validation_y = validation_data
        validation_X = np.asarray(validation_X)


        # -------------- code start : 结束 -------------
        # -------------- region start : 训练模型 -------------
        if self.verbose > 2 :
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('训练模型')
            print('训练模型')
        # -------------- code start : 开始 -------------
        # Initialize a Random Forest classifier with n trees
        forest = RandomForestClassifier(n_estimators=self.n_estimators,
                                        # min_samples_leaf = min_samples_leaf,
                                        # oob_score = True,
                                        max_features='log2',
                                        random_state=self.rand_seed)
        forest.fit(train_X, train_y)

        self.model = forest

        # -------------- code start : 结束 -------------
        if self.verbose > 2 :
            logging.debug('-' * 20)
            print('-' * 20)
        # -------------- region end : 训练模型 ---------------


    def predict(self, sentence,transform_input=False):
        '''
            预测一个句子的类别,对输入的句子进行预测

        :param sentences: 测试句子,
        :type sentences: str
        :param transform: 是否转换句子，如果为True,输入原始字符串句子即可，内部已实现转换成字典索引的形式。
        :type transform: bool
        '''
        y_pred = self.batch_predict([sentence],transform_input)[0]
        return y_pred

    def print_model_descibe(self):
        import pprint
        detail = {'rand_seed': self.rand_seed,
                  'verbose': self.verbose,
                  'n_estimators': self.n_estimators,
                  'min_samples_leaf': self.min_samples_leaf,
                  'vocabulary_size': self.feature_encoder.vocabulary_size,
                  'train_data_count': len(self.feature_encoder.train_data),
                  }
        pprint.pprint(detail)
        logging.debug(detail)
        return detail

    def accuracy(self, test_data,transform_input=False):
        '''
            预测,对输入的句子进行预测,并给出准确率
                1. 转换格式
                2. 批量预测
                3. 统计准确率等
                4. 统计F1(macro) :统计各个类别的F1值，然后进行平均

        :param test_data: 测试句子
        :type test_data: array-like
        :param transform_input:
        :type transform_input: bool
        :return: y_pred,is_correct,accu,f1
        :rtype:tuple
        '''
        # -------------- region start : 1. 转换格式 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('1. 转换格式')
            print('1. 转换格式')
        # -------------- code start : 开始 -------------

        test_X, test_y = test_data

        # -------------- code start : 结束 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
        # -------------- region end : 1. 转换格式 ---------------

        # -------------- region start : 2. 批量预测 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('2. 批量预测')
            print('2. 批量预测')
        # -------------- code start : 开始 -------------

        y_pred = self.batch_predict(test_X,transform_input)

        # -------------- code start : 结束 -------------
        if self.verbose > 1:
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


if __name__ == '__main__':
    train_X = ['你好', '无聊', '测试句子', '今天天气不错','我要买手机']
    trian_y = [1,3,2,2,3]
    test_X = ['句子','你好','你妹']
    test_y = [2,3,0]

    feature_encoder = FeatureEncoder(
                                     verbose=0,
                                     need_segmented=False,
                                     full_mode=True,
                                     remove_stopword=True,
                                     replace_number=True,
                                     lowercase=True,
                                     zhs2zht=True,
                                     remove_url=True,
                                     feature_method='bow',
                                     max_features=2000,
                                     )
    train_X_feature = feature_encoder.fit_transform(train_X)
    test_X_feature = feature_encoder.transform(test_X)
    print(train_X_feature)

    bow_rf = BowRandomForest(
        rand_seed=1337,
        verbose=0,
        n_estimators=200,
        min_samples_leaf=1,
        feature_encoder=feature_encoder,
    )
    # bow_rf.model_from_pickle('model.pkl')


    bow_rf.fit(train_data=(train_X_feature,trian_y),
               validation_data=(test_X_feature,test_y))
    print(bow_rf.batch_predict(test_X,transform_input=True))
    print(bow_rf.predict('你好',transform_input=True))
    bow_rf.accuracy((train_X_feature,trian_y),False)
    # bow_rf.accuracy((test_X_feature,test_y),False)
    bow_rf.print_model_descibe()

    bow_rf.save_model('model.pkl')
