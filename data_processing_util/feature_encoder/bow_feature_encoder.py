# encoding=utf8

__author__ = 'jdwang'
__date__ = 'create date: 2016-06-24'
__email__ = '383287471@qq.com'
import numpy as np
import pandas as pd
import logging
import timeit
from gensim.models import Word2Vec
from gensim.corpora.dictionary import Dictionary
from data_processing_util.jiebanlp.jieba_util import Jieba_Util
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


class FeatureEncoder(object):
    '''
        BOW特征编码器:基于sklearn的CountVectorizer,TfidfVectorizer实现，将句子转成 BOW（计算）或者TFIDF编码。
        包含以下主要函数：
            1. segment_sentence：对句子分词
            2. build_dictionary：构建字典
            3. sentence_to_index：将原始字符串句子转为字典索引列表
            4. sentence_padding：将句子补齐
            5. fit_transform：构建编码器并转换数据
            6. transform_sentence：对句子编码
            7. get_sentence_length：对句子长度计算
            8. print_sentence_length_detail： 打印训练库句子详情.
            9. print_model_descibe: 打印模型的详情.
            10. sentence_index_to_bow: 将索引转为onehot数据
            11. to_onehot_array： 生成训练库句子的onehot编码

    '''

    def __init__(self,
                 rand_seed=1337,
                 verbose=0,
                 need_segmented=True,
                 full_mode=True,
                 remove_stopword=True,
                 replace_number=True,
                 lowercase = True,
                 zhs2zht = True,
                 remove_url = True,
                 feature_method='bow',
                 max_features = None,


                 ):
        '''
            1. 初始化参数，并验证参数合法性
            2. build feature encoder

            :param rand_seed: 随机种子,假如设置为为None时,则随机取随机种子
            :type rand_seed: int
            :param verbose: 数值越大,输出越详细
            :type verbose: int
            :param need_segmented: 数据处理选项,是否需要经过分词处理;如果为False,那么输入的数据不需要分词,提供的数据的每个句子的每个词要以空格分割.比如: ['我 要 买 手机','你好','早上 好'];如果为True,提供原始输入句子即可,比如:['我要买手机','你好','早上好'].
            :type need_segmented: bool
            :param full_mode: jieba分词选项,是否使用 full mode,默认为True
            :type full_mode: bool
            :param remove_stopword: jieba分词选项,是否去除 stop word,默认为True
            :type remove_stopword: bool
            :param replace_number: jieba分词选项,是否将数据统一替换成NUM,默认为True
            :type replace_number: bool
            :param lowercase: jieba分词选项,是否将数据统一替换成NUM,默认为True
            :type lowercase: bool
            :param zhs2zht: jieba分词选项,出現繁体的時候，是否转简体,默认为True
            :type zhs2zht: bool
            :param remove_url: jieba分词选项,是否移除 微博url，http://t.cn/开头的地址,默认为True
            :type remove_url: bool
            :param feature_method: 模型设置选项,选择 bow或者tfidf 特征计算方法
            :type feature_method: str
            :param max_features: 模型设置选项,特征选择的最大特征词数
            :type max_features: int


        '''
        self.rand_seed = rand_seed
        self.verbose = verbose
        self.full_mode = full_mode
        self.remove_stopword = remove_stopword
        self.need_segmented = need_segmented
        self.replace_number = replace_number
        self.lowercase = lowercase
        self.zhs2zht = zhs2zht
        self.remove_url = remove_url
        self.feature_method = feature_method
        self.max_features = max_features



        # 检验参数合法性
        assert self.feature_method in ['bow','tfidf'],'feature method 只能取: bow,tfidf'


        # 原始训练数据
        self.train_data =None
        # 特征编码器:
        self.feature_encoder=None
        # 初始化jieba分词器
        if need_segmented:
            self.jieba_seg = Jieba_Util(verbose=self.verbose)
        # 切完词的句子
        self.segmented_sentences = None
        # 训练库提取出来的字典对象
        self.train_data_dict = None
        # 训练库提取出来的字典词汇列表
        self.vocabulary = None
        # 训练库提取出来的字典词汇个数
        self.vocabulary_size = None
        # 训练句子特征
        self.train_features = None

        # word2vec 模型
        # self.word2vec_model = None

        # self.fit_transform()

    def segment_sentence(self, sentence):
        '''
        对句子进行分词,使用jieba分词

        :param sentence: 句子
        :type sentence: str
        :return: 分完词句子，以空格连接
        :rtype: str
        '''

        segmented_sentence = self.jieba_seg.seg(sentence,
                                                sep=' ',
                                                full_mode=self.full_mode,
                                                remove_stopword=self.remove_stopword,
                                                replace_number=self.replace_number,
                                                lowercase = self.lowercase,
                                                zhs2zht= self.zhs2zht,
                                                remove_url=self.remove_url,
                                                )
        return segmented_sentence


    def fit_transform(self,train_data=None):
        '''
            build feature encoder
                1. 转换数据格式，并分词
                2. 构建vectorizer并拟合数据

        :param train_data: 训练句子列表:['','',...,'']
        :type train_data: array-like.
        :return: train_data 编码后的向量
        '''

        if self.verbose > 1:
            logging.debug('build feature encoder...')
            print 'build feature encoder...'

        # -------------- region start : 1. 转换数据格式，并分词 -------------
        if self.verbose > 2 :
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('1. 转换数据格式，并分词')
            print('1. 转换数据格式，并分词')
        # -------------- code start : 开始 -------------

        assert train_data is not None,'没有输入训练数据!'

        train_data = np.asarray(train_data)
        self.train_data = train_data

        if self.need_segmented:
            # 分词
            train_segmented_sentences = map(self.segment_sentence,train_data)
        else:
            train_segmented_sentences = train_data

        # -------------- code start : 结束 -------------
        if self.verbose > 2 :
            logging.debug('-' * 20)
            print('-' * 20)
        # -------------- region end : 1. 转换数据格式，并分词 ---------------

        if self.feature_method == 'tfidf':
            vectorizer = TfidfVectorizer(analyzer="word",
                                         token_pattern=u'(?u)\\b\w+\\b',
                                         tokenizer=None,
                                         preprocessor=None,
                                         lowercase=False,
                                         stop_words=None,
                                         # vocabulary = tfidf_vocabulary,
                                         max_features=self.max_features,
                                         )

        elif self.feature_method == 'bow':
            vectorizer = CountVectorizer(analyzer="word",
                                         token_pattern=u'(?u)\\b\w+\\b',
                                         tokenizer=None,
                                         preprocessor=None,
                                         lowercase=False,
                                         stop_words=None,
                                         # vocabulary = tfidf_vocabulary,
                                         max_features=self.max_features,
                                         )

        train_features = vectorizer.fit_transform(train_segmented_sentences).toarray()

        self.train_features = train_features
        self.feature_encoder = vectorizer
        self.vocabulary = vectorizer.get_feature_names()
        self.vocabulary_size = len(vectorizer.get_feature_names())

        return train_features

    def transform_sentence(self, sentence):
        '''
            转换一个句子的格式。跟训练数据一样的操作,对输入句子进行 bow或tfidf 编码。
                1. 分词
                2. 编码

        :param sentence: 输入句子,不用分词,进来后会有分词处理
        :type sentence: str
        :return: 补齐的字典索引
        :rtype: array-like
        '''



        # -------------- region start : 1. 分词 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print '-' * 20
            logging.debug('1. 分词')
            print '1. 分词'
        # -------------- code start : 开始 -------------

        # 分词
        if self.need_segmented:
            seg_sentence = self.segment_sentence(sentence)
        else:
            seg_sentence = sentence

        # -------------- code start : 结束 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print '-' * 20
        # -------------- region end : 1. 分词 ---------------

        # -------------- region start : 2. 转为字典索引列表,之后补齐,输入为补齐的字典索引列表 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print '-' * 20
            logging.debug('2. 转为字典索引列表,之后补齐,输入为补齐的字典索引列表')
            print '2. 转为字典索引列表,之后补齐,输入为补齐的字典索引列表'
        # -------------- code start : 开始 -------------

        features = self.feature_encoder.transform([seg_sentence]).toarray()[0]

        # -------------- code start : 结束 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print '-' * 20
        # -------------- region end : 2. 转为字典索引列表,之后补齐,输入为补齐的字典索引列表 ---------------

        return features

    def transform(self, data):
        '''
            批量转换数据，跟 transform_sentence()一样的操作
                1. 直接调用 self.transform_sentence 进行处理

        :param sentence: 输入句子
        :type sentence: array-like
        :return: bow vector
        :rtype: array-like
        '''

        index = map(self.transform_sentence, data)
        # print train_index[:5]

        return np.asarray(index)

    def print_model_descibe(self):
        '''
            打印模型参数详情

        :return: 参数设置详情
        :rtype: dict 或 {}
        '''
        import pprint
        detail = {'train_data_count': len(self.train_data),
                  'need_segmented': self.need_segmented,
                  'vocabulary_size': self.vocabulary_size,
                  'verbose': self.verbose,
                  'rand_seed': self.rand_seed,
                  'full_mode': self.full_mode,
                  'remove_stopword': self.remove_stopword,
                  'replace_number': self.replace_number,
                  'lowercase': self.lowercase,
                  'zhs2zht': self.zhs2zht,
                  'remove_url': self.remove_url,
                  'feature_method': self.feature_method,
                  'max_features': self.max_features,
                  }
        pprint.pprint(detail)
        logging.debug(detail)
        return detail

if __name__ == '__main__':
    train_data = ['你好，你好', '測試句子','无聊', '测试句子', '今天天气不错','买手机','你要买手机']
    test_data = ['你好，你好,si','无聊']
    feature_encoder = FeatureEncoder(rand_seed=1337,
                                     verbose=0,
                                     need_segmented=True,
                                     full_mode=True,
                                     remove_stopword=True,
                                     replace_number=True,
                                     lowercase = True,
                                     zhs2zht = True,
                                     remove_url = True,
                                     feature_method='bow',
                                     max_features = 100,
                                     )
    train_features = feature_encoder.fit_transform(train_data=train_data).toarray()
    print ','.join(feature_encoder.vocabulary)
    print train_features
    test_features = feature_encoder.transform(test_data)
    print test_features
    print feature_encoder.vocabulary_size
    feature_encoder.print_model_descibe()


