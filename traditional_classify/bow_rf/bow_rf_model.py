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
from commom.common_model_class import CommonModel
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


class BowRandomForest(CommonModel):
    '''
        使用BOW或者TFIDF作为特征输入，使用random forest作为分类器
    '''

    def __init__(self,
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
        :param feature_encoder: 输入数据的设置选项，设置输入编码器
        :type feature_encoder: onehot_feature_encoder.FeatureEncoder
        '''

        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf

        # random forest model
        self.model = None



    def model_from_pickle(self, path):
        pass

    def save_model(self, path):
        pass

    def batch_predict(self, sentence):

        y_pred = self.model.predict(test_tfidf_features)
        pass

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

        # Initialize a Random Forest classifier with n trees
        forest = RandomForestClassifier(n_estimators=self.n_estimators,
                                        # min_samples_leaf = min_samples_leaf,
                                        # oob_score = True,
                                        max_features='log2',
                                        random_state=0)
        forest.fit(train_X, train_y)

        self.model = forest

    def predict(self, sentence):
        y_pred = forest.predict(test_tfidf_features)
        print
        y_pred
        is_correct = y_pred == test_data['LABEL_INDEX']
        print(sum(is_correct))
        print(sum(is_correct) / (len(y_pred) * 1.0))

        pass

    def print_model_descibe(self):
        pass

    def accuracy(self, test_data):
        pass


if __name__ == '__main__':

    logging.debug('使用 %s 提取特征向量' % (config['model']))
    print('使用 %s 提取特征向量' % (config['model']))

    if config['model'] == 'tfidf':
        vectorizer = TfidfVectorizer(analyzer="word",
                                     token_pattern=u'(?u)\\b\w+\\b',
                                     tokenizer=None,
                                     preprocessor=None,
                                     lowercase=False,
                                     stop_words=None,
                                     # vocabulary = tfidf_vocabulary,
                                     max_features=config['max_keywords'])

    elif config['model'] == 'bow':
        vectorizer = CountVectorizer(analyzer="word",
                                     token_pattern=u'(?u)\\b\w+\\b',
                                     tokenizer=None,
                                     preprocessor=None,
                                     lowercase=False,
                                     stop_words=None,
                                     # vocabulary = tfidf_vocabulary,
                                     max_features=config['max_keywords']
                                     )

    bow_rf = BowRandomForest()