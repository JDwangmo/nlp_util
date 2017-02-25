# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-07-06'; 'last updated date: 2017-01-11'
    Email:   '383287471@qq.com'
    Describe: 一些经常用到的函数：
                1、transform_word2vec_model_name： 通过 名字获取 word2vec 模型名
                2、save_data: 保存数据成csv格式
                3、
"""
from __future__ import print_function

__author__ = 'jdwang'
__date__ = 'create date: 2016-07-06'
__email__ = '383287471@qq.com'
__version__ = '1.4'

from abc import ABCMeta, abstractmethod


class CommonModel(object):
    """
        模型的抽象父类，规范模型的方法

    """
    __metaclass__ = ABCMeta
    __version__ = '1.4'

    @abstractmethod
    def __init__(self,
                 rand_seed=1337,
                 verbose=0):
        self.rand_seed = rand_seed
        self.verbose = verbose

    @abstractmethod
    def fit(self, train_data=None, validation_data=None): pass

    @abstractmethod
    def predict(self, sentence): pass

    @abstractmethod
    def batch_predict(self, sentence): pass

    @abstractmethod
    def batch_predict_bestn(self, sentence, bestn=1): pass

    @staticmethod
    def get_feature_encoder(**kwargs): pass

    @abstractmethod
    def save_model(self, path): pass

    @abstractmethod
    def model_from_pickle(self, path): pass

    @abstractmethod
    def accuracy(self, test_data): pass

    @abstractmethod
    def print_model_descibe(self): pass
