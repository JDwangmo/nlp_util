# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-06-24'; 'last updated date: 2016-09-27'
    Email:   '383287471@qq.com'
    Describe: BOW feature encoder
"""
from __future__ import print_function

import numpy as np
import logging
from data_processing_util.word2vec_util.word2vec_util import Word2vecUtil
from data_processing_util.jiebanlp.jieba_util import Jieba_Util
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class FeatureEncoder(object):
    '''
        ## 简介
        BOW特征编码器:基于sklearn的CountVectorizer,TfidfVectorizer实现，将句子转成 BOW（计算）或者TFIDF编码。
        ## 目前支持两种粒度的切分： 字(word) 和 分词后的词(seg)
        包含以下主要函数：
            1. segment_sentence：对句子分词
            2. transform_sentence：buildin，对一个句子编码
            3. fit_transform：构建编码器并转换数据
            4. transform： 转换数据
            5. print_sentence_length_detail： todo,打印训练库句子详情.
            6. print_model_descibe: 打印模型的详情.

    '''

    def __init__(self,
                 # rand_seed=1337,
                 verbose=0,
                 need_segmented=True,
                 full_mode=True,
                 remove_stopword=True,
                 replace_number=True,
                 lowercase=True,
                 zhs2zht=True,
                 remove_url=True,
                 feature_method='bow',
                 feature_type='seg',
                 max_features=None,
                 word2vec_to_solve_oov=False,
                 save_middle_result=False,
                 **kwargs
                 ):
        '''
            1. 初始化参数，并验证参数合法性
            2. build feature encoder

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
            :param feature_type: 模型设置选项,选择不同粒度的特征单位， 目前只支持 word,seg和 word_seg。
                - word：直接以字为单位，比如 我要买手机--->我 要 买 手 机
                - seg：分词后的词单元为单位，比如 我要买手机--->我 要 买 手机
                - word_seg：分词后的字和词为单位，比如 我要买手机--->我 要 买 手机 手 机
            :type feature_type: str
            :param max_features: 模型设置选项,特征选择的最大特征词数
            :type max_features: int
            :param word2vec_to_solve_oov: 使用word2vec扩展oov词
            :type word2vec_to_solve_oov: bool
            :param save_middle_result: 是否保存中间结果，为了节约空间默认关闭！
            :type save_middle_result: bool
            :param kwargs: 支持 word2vec_model_file_path等
            :type kwargs: dict


        '''
        # self.rand_seed = rand_seed
        self.save_middle_result = save_middle_result
        self.verbose = verbose
        self.full_mode = full_mode
        self.remove_stopword = remove_stopword
        self.need_segmented = need_segmented
        self.replace_number = replace_number
        self.lowercase = lowercase
        self.zhs2zht = zhs2zht
        self.remove_url = remove_url
        self.feature_method = feature_method
        self.feature_type = feature_type
        self.max_features = max_features
        self.word2vec_to_solve_oov = word2vec_to_solve_oov
        self.kwargs = kwargs

        # 检验参数合法性
        assert self.feature_method in ['bow', 'tfidf'], 'feature method 只能取: bow,tfidf'
        assert self.feature_type in ['word', 'seg', 'word_seg'], 'feature type 只能取: word,seg和word_seg'

        if word2vec_to_solve_oov:
            # 加载word2vec模型
            if word2vec_to_solve_oov:
                assert kwargs.has_key('word2vec_model_file_path'), '请提供 属性 word2vec_model_file_path'
                # 加载word2vec模型
                w2v_util = Word2vecUtil()
                self.word2vec_model = w2v_util.load(kwargs.get('word2vec_model_file_path'))

        # 初始化jieba分词器
        if need_segmented:
            self.jieba_seg = Jieba_Util(verbose=self.verbose)

        # 特征编码器: bow or tf-idf transformer
        self.feature_encoder = None
        # 训练库提取出来的字典对象
        self.train_data_dict = None
        # 训练库提取出来的字典词汇列表
        self.vocabulary = None
        # 训练库提取出来的字典词汇个数
        self.vocabulary_size = None
        # 训练样例的个数
        self.train_data_count = 0

        # region 为了节约内存空间，实际运行中时，建议设置 save_middle_result = False（关闭中间结果的保存）
        if self.save_middle_result:
            # 原始训练数据
            self.train_data = None
            # 切完词的句子
            self.segmented_sentences = None
            # 训练句子特征
            self.train_features = None
            # endregion

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

        if self.feature_type == 'seg':
            segmented_sentence = self.jieba_seg.seg(
                sentence,
                sep=' ',
                full_mode=self.full_mode,
                remove_stopword=self.remove_stopword,
                replace_number=self.replace_number,
                lowercase=self.lowercase,
                zhs2zht=self.zhs2zht,
                remove_url=self.remove_url,
                HMM=False,
            )
        elif self.feature_type == 'word':
            # 将句子切分为 以字为单元 以空格分割
            # 1. 先使用jieba进行预处理，将数字替换等
            segmented_sentence = self.jieba_seg.iter_each_word(
                sentence,
                sep=' ',
                need_segmented=True,
                full_mode=self.full_mode,
                remove_stopword=self.remove_stopword,
                replace_number=self.replace_number,
                lowercase=self.lowercase,
                zhs2zht=self.zhs2zht,
                remove_url=self.remove_url,
                HMM=False,
            )
            # 2. 按字切分

        elif self.feature_type == 'word_seg':
            # 将句子切分为 以字和词为单元，相同则去重 以空格分割
            # 1. 先使用jieba进行预处理，将数字替换等
            segmented_sentence = self.jieba_seg.seg(
                sentence,
                sep=' ',
                full_mode=self.full_mode,
                remove_stopword=self.remove_stopword,
                replace_number=self.replace_number,
                lowercase=self.lowercase,
                zhs2zht=self.zhs2zht,
                remove_url=self.remove_url,
                HMM=False,
            )
            # print(segmented_sentence)
            # 2. 按字切分
            word = self.jieba_seg.iter_each_word(segmented_sentence, sep=' ', need_segmented=False).split()
            # 3. 按词切分
            seg = segmented_sentence.split()
            segmented_sentence = ' '.join(set(seg + word))
        else:
            assert False, '不支持其他粒度的切分！'

        return segmented_sentence

    def reset(self):
        """重置对象

        Returns
        -------

        """
        self.feature_encoder=None

    def fit_transform(self, train_data=None, test_data=None):
        """
            build feature encoder
                1. fit
                2. transform拟合数据

        :param train_data: 训练句子列表:['','',...,'']
        :type train_data: array-like.
        :return: train_data 编码后的向量
        """
        # 训练样例的个数
        self.train_data_count = len(train_data)

        return self.fit(train_data, test_data).transform(train_data)

    def fit(self, train_data=None, test_data=None):
        """
            build feature encoder
                1. 转换数据格式，并分词
                2. 构建vectorizer

        :param train_data: 训练句子列表:['','',...,'']
        :type train_data: array-like.
        :return: train_data 编码后的向量
        """

        if self.verbose > 1:
            logging.debug('build feature encoder...')
            print('build feature encoder...')

        # -------------- region start : 1. 转换数据格式，并分词 -------------
        if self.verbose > 2:
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('1. 转换数据格式，并分词')
            print('1. 转换数据格式，并分词')
        # -------------- code start : 开始 -------------

        assert train_data is not None, '没有输入训练数据!'

        train_data = np.asarray(train_data)
        # 为了节约内存空间，实际运行中时，建议设置 save_middle_result = False（关闭中间结果的保存）
        if self.save_middle_result:
            self.train_data = train_data

        if self.need_segmented:
            # 分词
            train_segmented_sentences = map(self.segment_sentence, train_data)
        else:
            # 不需要分词
            train_segmented_sentences = train_data

        # -------------- code start : 结束 -------------
        if self.verbose > 2:
            logging.debug('-' * 20)
            print('-' * 20)
        # -------------- region end : 1. 转换数据格式，并分词 ---------------
        if self.feature_encoder is None:
            # 当 feature_encoder 还没创建过时，则创建
            if self.feature_method == 'tfidf':
                self.feature_encoder = TfidfVectorizer(
                    analyzer="word",
                    token_pattern=u'(?u)\\b\w+\\b',
                    tokenizer=None,
                    preprocessor=None,
                    lowercase=False,
                    stop_words=None,
                    # vocabulary = tfidf_vocabulary,
                    max_features=self.max_features,
                )

            elif self.feature_method == 'bow':
                self.feature_encoder = CountVectorizer(
                    analyzer="word",
                    token_pattern=u'(?u)\\b\w+\\b',
                    tokenizer=None,
                    preprocessor=None,
                    lowercase=False,
                    stop_words=None,
                    # vocabulary = tfidf_vocabulary,
                    max_features=self.max_features,
                )
            else:
                raise NotImplementedError

        train_features = self.feature_encoder.fit_transform(train_segmented_sentences).toarray()

        # 为了节约内存空间，实际运行中时，建议设置 save_middle_result = False（关闭中间结果的保存）
        if self.save_middle_result:
            self.train_features = train_features

        # 字典
        self.vocabulary = self.feature_encoder.get_feature_names()
        # 字典个数
        self.vocabulary_size = len(self.vocabulary)

        return self

    def word_similarity(self, word2vec_model, word1, word2):
        '''
        计算两个词的相似性

        Parameters
        ----------
        word2vec_model : gensim object
            word2vec_model gensim Word2Vec model
        word2:
        word1:

        Returns
        --------
            similarity score: float
        '''
        try:
            return word2vec_model.n_similarity(word1, word2)
        except:
            return 0

    def replace_oov_with_similay_word(self, word2vec_model, sentence):
        '''
            对句子中oov词使用训练库中最相近的词替换（word2vec余弦相似性）

        :param sentence:
        :return:
        '''

        # is_oov = np.asarray([item for item in self.feature_encoder.vocabulary])
        # has_oov = any(is_oov)
        sentence = sentence.split()
        oov_word = []
        replace_word = []
        for item in sentence:
            if item not in self.vocabulary:
                oov_word.append(item)
                keywords_sim_score = np.asarray(
                    [self.word_similarity(word2vec_model, item, i) for i in self.vocabulary])
                sorted_index = np.argsort(keywords_sim_score)[-1::-1]
                most_similarity_score = keywords_sim_score[sorted_index[0]]
                most_similarity_word = self.vocabulary[sorted_index[0]]
                if self.verbose > 1:
                    print(u'%s 最相近的词是%s,分数为:%f' % (item, most_similarity_word, most_similarity_score))
                replace_word.append(most_similarity_word)
        sentence += replace_word
        return ' '.join(sentence)

    def transform_sentence(self,
                           sentence,
                           ):
        '''
            转换一个句子的格式。跟训练数据一样的操作,对输入句子进行 bow或tfidf 编码。
                1. 分词
                2. 编码

        :param sentence: 输入句子,不用分词,进来后会有分词处理
        :type sentence: str
        :return: 补齐的字典索引
        :rtype: array-like
        '''

        # region -------------- 1. 分词 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('1. 分词')
            print('1. 分词')
        # -------------- code start : 开始 -------------

        # 分词
        if self.need_segmented:
            seg_sentence = self.segment_sentence(sentence)
        else:
            seg_sentence = sentence

        if self.word2vec_to_solve_oov:
            seg_sentence = self.replace_oov_with_similay_word(self.word2vec_model,
                                                              seg_sentence)

        # -------------- code start : 结束 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
        # endregion -------------- 1. 分词 ---------------

        # region -------------- 2. 特征转换 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('2. 转为字典索引列表,之后补齐,输入为补齐的字典索引列表')
            print('2. 转为字典索引列表,之后补齐,输入为补齐的字典索引列表')
        # -------------- code start : 开始 -------------

        features = self.feature_encoder.transform([seg_sentence]).toarray()[0]

        # -------------- code start : 结束 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
        # endregion -------------- 2. 特征转换 ---------------

        return features

    def transform(self,
                  data,
                  ):
        '''
            批量转换数据，跟 transform_sentence()一样的操作
                1. 直接调用 self.transform_sentence 进行处理

        :param data: 输入句子集合
        :type data: array-like
        :return: bow vector
        :rtype: array-like
        '''

        index = map(self.transform_sentence, data)
        # print(index[:5])

        return np.asarray(index)

    def print_model_descibe(self):
        '''
            打印模型参数详情

        :return: 参数设置详情
        :rtype: dict 或 {}
        '''
        import pprint
        detail = {'train_data_count': self.train_data_count,
                  'need_segmented': self.need_segmented,
                  'word2vec_to_solve_oov': self.word2vec_to_solve_oov,
                  'vocabulary_size': self.vocabulary_size,
                  'verbose': self.verbose,
                  # 'rand_seed': self.rand_seed,
                  'full_mode': self.full_mode,
                  'remove_stopword': self.remove_stopword,
                  'replace_number': self.replace_number,
                  'lowercase': self.lowercase,
                  'zhs2zht': self.zhs2zht,
                  'remove_url': self.remove_url,
                  'feature_method': self.feature_method,
                  'feature_type': self.feature_type,
                  'max_features': self.max_features,
                  }
        pprint.pprint(detail)
        logging.debug(detail)
        return detail


def test_word_bow_feature():
    feature_encoder = FeatureEncoder(
        verbose=0,
        need_segmented=True,
        full_mode=True,
        remove_stopword=True,
        replace_number=True,
        lowercase=True,
        zhs2zht=True,
        remove_url=True,
        feature_method='bow',
        feature_type='word',
        max_features=100,
    )
    train_features = feature_encoder.fit_transform(train_data=train_data)
    print(','.join(feature_encoder.vocabulary))
    print(train_features)
    test_features = feature_encoder.transform(test_data)
    print(test_features)
    print(feature_encoder.vocabulary_size)
    feature_encoder.print_model_descibe()


def test_seg_bow_feature():
    feature_encoder = FeatureEncoder(
        verbose=0,
        need_segmented=True,
        full_mode=True,
        remove_stopword=True,
        replace_number=True,
        lowercase=True,
        zhs2zht=True,
        remove_url=True,
        feature_method='bow',
        feature_type='seg',
        max_features=100,
    )
    train_features = feature_encoder.fit_transform(train_data=train_data)
    print(','.join(feature_encoder.vocabulary))
    print(train_features)
    test_features = feature_encoder.transform(test_data)
    print(test_features)
    print(feature_encoder.vocabulary_size)
    feature_encoder.print_model_descibe()


def test_word_seg_bow_feature():
    feature_encoder = FeatureEncoder(
        verbose=0,
        need_segmented=True,
        full_mode=True,
        remove_stopword=True,
        replace_number=True,
        lowercase=True,
        zhs2zht=True,
        remove_url=True,
        feature_method='bow',
        feature_type='word_seg',
        max_features=100,
    )
    train_features = feature_encoder.fit_transform(train_data=train_data)
    print(','.join(feature_encoder.vocabulary))
    # print(train_features)
    test_features = feature_encoder.transform(test_data)
    print(test_features)
    print(feature_encoder.vocabulary_size)
    # feature_encoder.print_model_descibe()


if __name__ == '__main__':
    train_data = ['你好，你好', '測試句子', '无聊', '测试句子', '今天天气不错', '买手机', '50元', '妈B', '你要买手机', 'ch2r']
    test_data = ['你好，你好,si', '无聊']
    # 测试字的bow向量编码
    test_word_bow_feature()
    # 测试词的bow向量编码
    # test_seg_bow_feature()
    # 测试以字和词为单位的向量编码
    # test_word_seg_bow_feature()
