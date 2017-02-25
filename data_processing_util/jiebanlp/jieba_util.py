# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-05-15'; 'last updated date: 2017-01-10'
    Email:   '383287471@qq.com'
    Describe: Onehot Encoder --- Onehot特征编码器,将句子转成 onehot编码
"""
__author__ = 'jdwang'
__version__ = '1.4'

import numpy as np
import logging
import timeit
import jieba
import jieba.posseg as jseg
import os
import re
import io
import opencc


class Jieba_Util(object):
    """
        the wrapper of jieba tool.
        对jieba分词进行一层包装.函数有：
            1. convert_to_simple_chinese： 转换为简体中文
            2. seg： 中文分词

    """
    __version__ = '1.4'

    def __init__(self,
                 num_of_parallel=10,
                 verbose=0):
        """
            1. 初始化参数
            2. 加载用户字典和stop word列表

        Parameters
        ----------
        num_of_parallel : int
            并行的线程数
        verbose: int
            数值越大，打印越多的详细信息，设置为0时，什么信息都不显示.
        """
        # 初始化参数
        self.verbose = verbose
        # 设置jieba分词对线程
        jieba.enable_parallel(num_of_parallel)

        # -------------- region start : 2. 加载用户字典和stop word列表 -------------
        if verbose > 1:
            logging.debug('-' * 20)
            print '-' * 20
            logging.debug('2. 加载用户字典和stop word列表')
            print '2. 加载用户字典和stop word列表'
        # -------------- code start : 开始 -------------


        # region 添加用户字典
        jieba.load_userdict(os.path.dirname(__file__) + '/userdict.txt')
        # 添加 261,529 个用户词典（来自在线新华词典的抓取）
        jieba.load_userdict(os.path.dirname(__file__) + '/vocabulary_len2_xiandaihanyu.txt')
        # endregion
        self.stopword_list = io.open(os.path.dirname(__file__) + '/stopword.txt', 'r',
                                     encoding='utf8').read().strip().split()
        self.exclude_word_list = set(['886', '88'])

        # -------------- code start : 结束 -------------
        if verbose > 1:
            logging.debug('-' * 20)
            print '-' * 20
            # -------------- region end : 2. 加载用户字典和stop word列表 ---------------

    def pos_seg(self, word):

        # for items in jseg.lcut(word):
        #     print items.flag,items.word

        return jseg.lcut(word)[0].flag

    def convert_to_simple_chinese(self, sentence):
        '''
            将句子由繁体转为简体

        :param sentence: 原始句子
        :type sentence: str
        :return: 简体中文句子
        :rtype: str
        '''
        simple_chinese = opencc.convert(sentence, config='zht2zhs.ini')
        return simple_chinese

    def iter_each_word(self, sentence, sep=' ', need_segmented=True, **kwargs):
        '''
            返回句子的每个词（中文字或英文单词），比如
                - 500 元  --> [500,元]
                - NUMBER 元  --> [NUMBER,元]
                - ch2r 你好  --> [ch2r,你，好]
            注意以空格分割词，如果还未分词，可以设置 need_segmented为True

        :param sentence:
        :param need_segmented: 是否需要先进行分词处理，默认为True
        :param kwargs: full_mode,remove_stopword,replace_number,lowercase,zhs2zht,remove_url,HMM
        :return:
        '''

        if need_segmented:
            sentence = self.seg(
                sentence,
                sep=' ',
                full_mode=kwargs.get('full_mode', True),
                remove_stopword=kwargs.get('remove_stopword', True),
                replace_number=kwargs.get('replace_number', True),
                lowercase=kwargs.get('lowercase', True),
                zhs2zht=kwargs.get('zhs2zht', True),
                remove_url=kwargs.get('remove_url', True),
                HMM=kwargs.get('HMM', False),
            )
        words = []
        for item in sentence.split():
            if re.findall(u'[0-9A-Za-z]+', item):
                words.append(item)
            else:
                if type(item) is not unicode:
                    item = item.decode('utf8')

                word = list(item)
                words.extend(word)
        words = sep.join(words)
        return words

    def seg(self,
            sentence,
            sep=' ',
            full_mode=False,
            remove_stopword=False,
            replace_number=False,
            lowercase=True,
            zhs2zht=True,
            remove_url=True,
            HMM=False,
            ):
        """
            使用 jieba 分词进行分词

        :param sentence: 待分词句子
        :type sentence: str
        :param sep: 将句子分完词之后使用什么字符连接，默认以空格连接.
        :type sep: str
        :param full_mode: jieba设置选项，是否使用full mode分词模式.
        :type full_mode: bool
        :param remove_stopword: 是否去除 stop word
        :type remove_stopword: bool
        :param replace_number: 是否把数字统一替换成字符 NUM
        :type replace_number: bool
        :param lowercase: 是否把字母转成小写
        :type lowercase: bool
        :param zhs2zht: 出現繁体的時候，是否转简体
        :type zhs2zht: bool
        :param remove_url: 是否移除 微博url，包含t.cn的url，比如：http://t.cn/开头的地址或者//t.cn/R50TdMg
        :type remove_url: bool
        :param HMM: 是否启用HMM发现新词模式，默认为False
        :type HMM: bool
        :return: 返回分词后字符串,seg_srt
        :rtype: str

        """
        # 先去除所有空格
        sentence = sentence.replace(' ', '')

        if lowercase:
            # 转成小写
            sentence = sentence.lower()
        if zhs2zht:
            # 繁体转简体
            sentence = self.convert_to_simple_chinese(sentence)
        if remove_url:
            # sentence = re.sub(u'(http:)//t.cn/[a-zA-Z0-9]*$', '', sentence)
            sentence = re.sub(u'(http:|)//t.cn/[a-zA-Z0-9]+', '', sentence)

        # 数字对模式匹配
        num_pattern = re.compile('[0-9][0-9\.]*$')
        words = []
        for item in jieba.lcut(sentence, HMM=False):
            if num_pattern.match(item):
                # 匹配上数字
                if not replace_number:
                    words.append(item)
                elif item not in self.exclude_word_list:
                    word = num_pattern.sub('NUMBER', item)
                    words.append(word)
                    if self.verbose > 1:
                        logging.debug(u'句子（%s）将数字："%s" 替换成标记："NUMBER"' % (sentence, item))
                        print(u'句子（%s）将数字："%s" 替换成标记："NUMBER"' % (sentence, item))
                else:
                    words.append(item)

            elif remove_stopword and item in self.stopword_list:
                # 移除 stop words
                if self.verbose > 1:
                    logging.debug(u'句子（%s）去除stopwords：%s' % (sentence, item))
            else:
                # 其他词如果词性是 x， 则识别到标点符号
                is_x = False
                for word, pos in jseg.lcut(item, HMM=HMM):
                    # print word,pos
                    if pos in ['x']:
                        is_x = True
                        # words.append(word)

                if is_x:
                    # 标点符号
                    # print item
                    if self.verbose > 1:
                        logging.debug(u'句子（%s）将标点符号："%s"替换成""' % (sentence, ''))
                else:
                    words.append(item)

        sentence = ' '.join(words)
        # print sentence
        # print sentence
        seg_list = jieba.lcut(sentence, cut_all=full_mode)
        # print seg_list
        seg_list = [item for item in seg_list if len(item.strip()) != 0]
        # print seg_list
        seg_srt = sep.join(seg_list)

        return seg_srt


if __name__ == '__main__':
    # 使用样例
    jieba_util = Jieba_Util(verbose=0)
    sent = u'我喜歡买手机啊!!........'
    # sent = u'測試句子...'
    # sent = u'这手机,好用吗?'
    # sent = u'睡了。//t.cn/R50TdMgn你好'
    # sent = u'睡了。http://t.cn/R50TdMgn你好'
    # sent = u'2 b 的 2 0 0 元 。 不 想 买      了  。'
    # sent = u'哪台手机好'
    # sent = u'哪款好'
    sent = u'没事了'
    sent = u'拜拜'
    sent = u'妈B'
    sent = u'我不知道买哪款好'
    sent = u'见让坐。'
    sent = u'云计算'

    # print seg(sent,sep='|',full_mode=False,remove_stopword=True)
    # sent = u'有哪些1000块的手机适合我'
    # print seg(sent,sep='|',full_mode=False,remove_stopword=True)
    # sent = u'妈 B'
    # sent = u'2000元'
    # sent = u'2000元'
    # print ','.join(jieba.cut(sent,HMM=True))
    # print ','.join(jieba.cut(sent,HMM=False))
    # print jieba_util.iter_each_word('你 是 谁', sep=' ',need_segmented=False)
    print jieba_util.iter_each_word(u'妈b', sep=' ', need_segmented=True, full_mode=False)
    print jieba_util.iter_each_word('ch2r', sep=' ', need_segmented=False)
    print(jieba_util.seg(sent,
                         full_mode=True,
                         remove_stopword=False,
                         replace_number=True,
                         lowercase=True,
                         zhs2zht=True,
                         remove_url=False,
                         HMM=True
                         ))
    print(jieba_util.seg(sent,
                         sep='|',
                         full_mode=False,
                         remove_stopword=True,
                         replace_number=False,
                         HMM=False
                         ))
    # print jieba_util.seg(sent, sep='|', full_mode=False)
