#encoding=utf8
__author__ = 'jdwang'
__date__ = 'create date: 2016-05-15'
import numpy as np
import logging
import timeit
import jieba
import jieba.posseg as jseg
import os
import re
import io

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)





class Jieba_Util(object):
    def __init__(self,
                 verbose = 0):
        '''
            1. 初始化参数
            2. 加载用户字典和stop word列表
        '''
        self.verbose = verbose


        jieba.load_userdict(os.path.dirname(__file__) + '/userdict.txt')
        logging.debug('=' * 20)
        logging.debug('加载stopwords列表....')
        self.stopword_list = io.open(os.path.dirname(__file__) + '/stopword.txt', 'r',
                                     encoding='utf8').read().strip().split()
        logging.debug(u'stopwords有：%s' % (','.join(self.stopword_list)))
        logging.debug('=' * 20)
        self.exclude_word_list = set(['886', '88'])
        logging.debug(u'exclude words有：%s' % (','.join(self.exclude_word_list)))
        logging.debug('=' * 20)
        jieba.enable_parallel(10)


    def pos_seg(self,word):

        # for items in jseg.lcut(word):
        #     print items.flag,items.word

        return jseg.lcut(word)[0].flag

    def seg(self,
            sentence,sep='|',
            full_mode = True,
            remove_stopword = False,
            replace_number = True,
            verbose = 2):
        '''
        使用jieba分词进行分词
        :param sentence: 待分词句子
        :type sentence: str
        :param remove_stopword: 是否去除stopword
        :type remove_stopword: bool
        :param replace_number: 是否把数字替换成字符 NUM
        :type replace_number: bool
        :return:返回分词后字符串,seg_srt
        :rtype: str
        '''
        # logging.debug('是否去除stopwords：%s'%remove_stopword)
        # for items in jseg.lcut(sentence):
        #     print items.flag,items.word


        seg = []
        pattern = re.compile('[0-9]+$')
        for items in jseg.lcut(sentence):
            # print items.flag
            # 利用词性标注去除标点符号
            if items.flag in ['x']:
                logging.debug(u'句子（%s）将标点符号："%s"替换成""'%(sentence,items.word))
                seg.append('')
                # continue
            elif remove_stopword and items.word in self.stopword_list:
                logging.debug(u'句子（%s）去除stopwords：%s' % (sentence,items))
                continue
            # 将数字替换成 NUM
            elif pattern.match(items.word) and items.word not in self.exclude_word_list:
                if replace_number:
                    if verbose>1:
                        print(u'句子（%s）将数字："%s" 替换成标记："NUMBER"'%(sentence,items.word))
                    seg.append('NUMBER')
                    logging.debug(u'句子（%s）将数字："%s" 替换成标记："NUMBER"'%(sentence,items.word))
                else:
                    seg.append(items.word)
            else:
                seg.append(items.word)
        # sentence = [items.word for items in jseg.lcut(sentence) if items.flag!='x']


        sentence = ' '.join(seg)
        # print sentence
        # print sentence
        seg_list = jieba.lcut(sentence, cut_all=full_mode)
        # print seg_list
        seg_list = [item for item in seg_list if len(item.strip())!=0]
        # print seg_list
        seg_srt = sep.join(seg_list)
        return seg_srt



if __name__ == '__main__':
    sent = u'买手机啊........'
    # sent = u'k好,'
    # print seg(sent,sep='|',full_mode=False,remove_stopword=True)
    # sent = u'有哪些1000块的手机适合我'
    # print seg(sent,sep='|',full_mode=False,remove_stopword=True)
    # sent = u'妈B'
    print ','.join(jieba.cut(sent))
    print seg(sent,sep='|',full_mode=False)
