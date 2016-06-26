#encoding=utf8
__author__ = 'jdwang'
__date__ = 'create date: 2016-06-24'
__email__ = '383287471@qq.com'
import numpy as np
import pandas as pd
import logging
import timeit
from gensim.corpora.dictionary import Dictionary
from data_processing_util.jiebanlp.jieba_util import Jieba_Util

class FeatureEncoder(object):
    '''
    输入的特征编码器,将句子转成onehot编码(以字典索引形式表示)
    '''
    def __init__(self,
                 train_data=None,
                 need_segmented = True,
                 verbose=0,
                 full_mode=True,
                 remove_stopword=True,
                 sentence_padding_length = 7,
                 ):
        '''
            1. 初始化参数
            2. build feature encoder

        :param train_data: 训练句子列表:[[],[],...,[]]
        :type train_data: array-like.
        :param need_segmented: 数据处理选项,是否需要经过分词处理;如果为False,那么输入的数据不需要分词,提供的数据的每个句子的每个词要以空格分割.比如: ['我 要 买 手机','你好','早上 好'];如果为True,提供原始输入句子即可,比如:['我要买 手机','你好','早上好'].
        :type need_segmented: bool
        :param verbose: 数值越大,输出越详细
        :type verbose: int
        :param full_mode: jieba分词选项,是否使用 full mode,默认为True
        :type full_mode: bool
        :param remove_stopword: jieba分词选项,是否去除 stop word,默认为True
        :type remove_stopword: bool

        '''
        self.full_mode = full_mode
        self.remove_stopword = remove_stopword
        self.verbose = verbose
        self.sentence_padding_length = sentence_padding_length
        self.train_data = train_data
        self.need_segmented = need_segmented

        # 初始化jieba分词器
        self.jieba_seg = Jieba_Util(verbose=self.verbose)
        # 切完词的句子
        self.segmented_sentences = None
        # 训练库提取出来的字典
        self.train_data_dict = None
        # 训练库提取出来的字典词汇个数
        self.train_data_dict_size=None
        # 训练库句子的字典索引形式
        self.train_index = None
        # 训练库句子的补齐的字典索引形式
        self.train_padding_index = None

        if verbose > 1:
            logging.debug('build feature encoder...')
            print 'build feature encoder...'
        self.build_encoder()

    def segment_sentence(self, sentence):
        '''
        对句子进行分词,使用jieba分词
        :param sentence: 句子
        :type sentence: str
        :return:
        '''
        segmented_sentence = self.jieba_seg.seg(sentence,
                                                sep=' ',
                                                full_mode=self.full_mode,
                                                remove_stopword=self.remove_stopword,
                                                )
        return segmented_sentence

    def build_dictionary(self):
        '''
            1.对数据进行分词
            2.构建训练库字典,插入 一个特殊字符 'UNKOWN'表示未知词
        :return:
        '''

        # -------------- region start : 1.对数据进行分词 -------------
        if self.verbose > 1 :
            logging.debug('-' * 20)
            print '-' * 20
            logging.debug('对数据进行分词')
            print '对数据进行分词'
        # -------------- code start : 开始 -------------
        if self.need_segmented:
            self.segmented_sentences = map(self.segment_sentence, self.train_data)
        else:
            self.segmented_sentences = self.train_data
            # -------------- code start : 结束 -------------
        if self.verbose > 1 :
            logging.debug('-' * 20)
            print '-' * 20
        # -------------- region end : 1.对数据进行分词 ---------------
        # -------------- region start : 2.构建训练库字典 -------------
        if self.verbose > 1 :
            logging.debug('-' * 20)
            print '-' * 20
            logging.debug('2.构建训练库字典')
            print '2.构建训练库字典'
        # -------------- code start : 开始 -------------

        logging.debug('=' * 20)
        logging.debug('首先,构建训练库字典,然后将每个词映射到一个索引,再将所有句子映射成索引的列表')

        # 将训练库所有句子切分成列表,构成 2D的训练文档,每个单元是一个token,
        # 比如: [['今年','你','多少岁'],['你', '二十四','小时','在线','吗'],...]
        # 将分完词句子转成合适的数据格式
        train_document = map(lambda x: x.split(), self.segmented_sentences)
        # 获取训练库字典
        self.train_data_dict = Dictionary.from_documents(train_document)

        # 更新字典,再字典中添加特殊符号,其中
        # UNKOWN表示未知字符,即OOV词汇
        self.train_data_dict.add_documents([[u'UNKOWN']])

        self.train_data_dict_size = len(self.train_data_dict.keys())

        # -------------- print start : just print info -------------
        if self.verbose > 1 :
            logging.debug('训练库字典为:%d' % (len(self.train_data_dict.keys())))
            print '训练库字典为:%d' % (len(self.train_data_dict.keys()))
            logging.debug(u'字典有:%s' % (','.join(self.train_data_dict.token2id.keys())))
            print u'字典有:%s' % (','.join(self.train_data_dict.token2id.keys()))
        # -------------- print end : just print info -------------

        # -------------- code start : 结束 -------------
        if self.verbose > 1 :
            logging.debug('-' * 20)
            print '-' * 20
        # -------------- region end : 2.构建训练库字典 ---------------


    def sentence_to_index(self,sentence):
        """
            将 sentence 转换为 index,如果 token为OOV词,则分配为 UNKOWN
        :type sentence: str
        :param sentence: 以空格分割
        :return:
        """
        unknow_token_index = self.train_data_dict.token2id[u'UNKOWN']
        # 将训练库中所有句子的每个词映射到索引上,变成索引列表
        # 注意这里把所有索引都加1,目的是为了保留 索引0(用于补充句子),在神经网络上通过mask_zero忽略,实现变长输入
        index = [self.train_data_dict.token2id.get(item, unknow_token_index) + 1 for item in sentence.split()]
        return index

    def sentence_padding(self,sentence):
        '''
        将不等长的句子都对齐,超出padding_length长度的句子截断,小于的则补0
        :type sentence: list
        :param sentence: 以索引列表表示的句子
        :type padding_length: int
        :param padding_length: 补齐长度
        :return:
        '''

        padding_length = self.sentence_padding_length
        # print sentence
        sentence_length = len(sentence)
        if sentence_length > padding_length:
            # logging.debug(u'对句子进行截断:%s' % (sentence))

            sentence = sentence[:padding_length]

            # logging.debug(u'对句子进行截断后:%s' % (' '.join(seg[:padding_length])))
            # print(u'对句子进行截断后:%s' % (' '.join(seg[:padding_length])))
        elif sentence_length < padding_length:
            should_padding_length = padding_length - sentence_length
            left_padding = np.asarray([0] * (should_padding_length / 2))
            right_padding = np.asarray([0] * (should_padding_length - len(left_padding)))
            sentence = np.concatenate((left_padding, sentence, right_padding), axis=0)

        return sentence

    def build_encoder(self):
        '''
            build feature encoder
                1. 构建训练库字典
                2. 将训练句子转成字典索引形式
                3. 将句子补齐到等长,补齐长度为: self.sentence_padding_length
        :return:
        '''
        logging.debug('=' * 20)
        if self.train_data is None:
            logging.debug('没有输入训练数据!')
            print '没有输入训练数据!'
            quit()


        # -------------- region start : 1.构建训练库字典 -------------
        if self.verbose > 1 :
            logging.debug('-' * 20)
            print '-' * 20
            logging.debug('1.构建训练库字典')
            print '1.构建训练库字典'
        # -------------- code start : 开始 -------------

        # 构建训练库字典
        self.build_dictionary()

        # -------------- code start : 结束 -------------
        if self.verbose > 1 :
            logging.debug('-' * 20)
            print '-' * 20
        # -------------- region end : 1.构建训练库字典 ---------------

        # -------------- region start : 2. 将训练句子转成字典索引形式 -------------
        if self.verbose > 1 :
            logging.debug('-' * 20)
            print '-' * 20
            logging.debug('2. 将训练句子转成字典索引形式')
            print '2. 将训练句子转成字典索引形式'
        # -------------- code start : 开始 -------------

        # 将训练库中所有句子的每个词映射到索引上,变成索引列表
        self.train_index = map(self.sentence_to_index, self.segmented_sentences)
        # print train_index[:5]

        # -------------- code start : 结束 -------------
        if self.verbose > 1 :
            logging.debug('-' * 20)
            print '-' * 20
        # -------------- region end : 2. 将训练句子转成字典索引形式 ---------------

        # -------------- region start : 3. 将句子补齐到等长 -------------
        if self.verbose > 1 :
            logging.debug('-' * 20)
            print '-' * 20
            logging.debug('3. 将句子补齐到等长')
            print '3. 将句子补齐到等长'
        # -------------- code start : 开始 -------------

        # 将不等长的句子都对齐,超出padding_length长度的句子截断,小于的则补0
        train_padding_index = np.asarray(map(self.sentence_padding, self.train_index))
        self.train_padding_index = train_padding_index

        # -------------- code start : 结束 -------------
        if self.verbose > 1 :
            logging.debug('-' * 20)
            print '-' * 20
        # -------------- region end : 3. 将句子补齐到等长 ---------------





    def encoding_sentence(self,sentence):
        '''
            跟训练数据一样的操作,对输入句子进行 padding index 编码,将sentence转为补齐的字典索引
                1. 分词
                2. 转为字典索引列表,之后补齐,输入为补齐的字典索引列表
        :param sentence: 输入句子,不用分词,进来后会有分词处理
        :type sentence: str
        :return: 补齐的字典索引
        :rtype: array-like
        '''

        # -------------- region start : 1. 分词 -------------
        if self.verbose > 1 :
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
        if self.verbose > 1 :
            logging.debug('-' * 20)
            print '-' * 20
        # -------------- region end : 1. 分词 ---------------

        # -------------- region start : 2. 转为字典索引列表,之后补齐,输入为补齐的字典索引列表 -------------
        if self.verbose > 1 :
            logging.debug('-' * 20)
            print '-' * 20
            logging.debug('2. 转为字典索引列表,之后补齐,输入为补齐的字典索引列表')
            print '2. 转为字典索引列表,之后补齐,输入为补齐的字典索引列表'
        # -------------- code start : 开始 -------------

        sentence_index = self.sentence_to_index(seg_sentence)
        sentence_padding_index = self.sentence_padding(sentence_index)

        # -------------- code start : 结束 -------------
        if self.verbose > 1 :
            logging.debug('-' * 20)
            print '-' * 20
        # -------------- region end : 2. 转为字典索引列表,之后补齐,输入为补齐的字典索引列表 ---------------

        return sentence_padding_index


if __name__ == '__main__':
    train_data = ['你好','无聊','测试句子','今天天气不错']
    test_data = '句子'
    feature_encoder = FeatureEncoder(train_data=train_data,
                                     verbose=0)
    print feature_encoder.train_padding_index
    print feature_encoder.encoding_sentence(test_data)