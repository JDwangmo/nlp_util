# encoding=utf8
__author__ = 'jdwang'
__date__ = 'create date: 2016-07-02'
__email__ = '383287471@qq.com'
import numpy as np
import pandas as pd
import logging
import timeit
from gensim.models import Word2Vec
import io


class Word2vecUtil(object):
    '''
        在gensim.Word2Vec基础上封装一层，包含的函数有：
            1. train：训练word2vec模型
            2. save： 保存模型文件
            3. load：加载word2vec模型
    '''

    def __init__(self,
                 verbose = 0,
                 size=50,
                 train_method='cbow',
                 iter=5,
                 ):
        '''
            初始化参数,并检验参数的合法性

        :param verbose: 数值越大，输出越详细的信息
        :type verbose: int
        :param size: word2vec设置参数，词向量维度 ，默认为50维
        :type size: int
        :param train_method: word2vec设置参数，词向量训练方法，有两种（['cbow','skip']），默认使用cbow，
        :type train_method: str
        :param iter: word2vec设置参数，词向量训练迭代次数，默认为5
        :type iter: int
        '''
        self.verbose = verbose
        self.size = size
        self.train_method = train_method
        self.iter = iter

        # 检验参数的合法性
        assert train_method in ['cbow', 'skip'], 'train method 只能取 cbow 或者 skip'

    class MySentences(object):
        def __init__(self, sentence):
            self.sentence = sentence
            pass

        def __iter__(self):
            count = 0
            for line in self.sentence:
                count += 1
                if count % 100 == 0:
                    print (count + 1), line
                    logging.debug((count + 1))
                yield line.split(' ')

    def train(self, sentences):
        '''
            训练word2vec模型

        :param sentences: 句子列表，句子已经分好词，内部以空格分割词
        :return:
        '''

        sentences = [item.split() for item in sentences]
        # sentences = self.MySentences(sentences)
        if self.train_method == 'cbow':
            sg = 0
        elif self.train_method == 'skip':
            sg = 1

        self.model = Word2Vec(sentences=sentences,
                              size=self.size,
                              min_count=0,
                              workers=3,
                              alpha=0.025,
                              window=10,
                              max_vocab_size=None,
                              sample=1e-3,
                              seed=1,
                              min_alpha=0.0001,
                              sg=sg,
                              hs=0,
                              negative=5,
                              cbow_mean=1,
                              hashfxn=hash,
                              iter=self.iter,
                              null_word=0,
                              trim_rule=None,
                              sorted_vocab=1,
                              # batch_words=MAX_WORDS_IN_BATCH
                              )
        # -------------- print start : just print info -------------
        if self.verbose > 1 :
            logging.debug('字典大小为：%d'%len(self.model.vocab))
            print '字典大小为：%d'%len(self.model.vocab)
        # -------------- print end : just print info -------------

    def save(self, output_path):
        self.model.save(output_path)

    def load(self, input_path):
        self.model = Word2Vec.load(input_path)
        return self.model

    def print_model_descibe(self):
        '''
            打印模型参数详情

        :return: 参数设置详情
        :rtype: dict 或 {}
        '''
        import pprint
        detail = {
            'verbose': self.verbose,
            'vector_size': self.size,
            'train_method': self.train_method,
            'iter': self.iter,
            'vocab size': len(self.model.vocab),
        }
        pprint.pprint(detail)
        logging.debug(detail)
        return detail


def test():
    input_file1 = './sample_data/train_data_half_2090.csv'

    data = pd.read_csv(input_file1,
                       encoding='utf8',
                       sep='\t',
                       header=0)
    sentences = data['WORDS'].as_matrix()
    print '句子数：%d' % sentences.shape

    util = Word2vecUtil(size=50,
                        train_method='skip'
                        )
    util.train(sentences)
    util.print_model_descibe()

    most_similar_words = util.model.most_similar(u'喜欢')
    print ','.join([i for i, j in most_similar_words])
    util.save('vector/train_data_half_2090.gem')


if __name__ == '__main__':
    test()
