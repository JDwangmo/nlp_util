# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-07-02','last update date: 2016-09-02'
    Email:   '383287471@qq.com'
    Describe:
"""
import logging
import pandas as pd
from gensim.models import Word2Vec
from data_processing_util.jiebanlp.jieba_util import Jieba_Util
import os


class Word2vecUtil(object):
    '''
        在gensim.Word2Vec基础上封装一层，包含的函数有：
            1. train：训练word2vec模型
            2. save： 保存模型文件
            3. load：加载word2vec模型
    '''

    def __init__(self,
                 verbose=0,
                 size=50,
                 train_method='cbow',
                 iter=5,
                 ):
        """
            初始化参数,并检验参数的合法性

        Parameters
        ----------
        verbose: int
            数值越大，输出越详细的信息
        size:int
            word2vec设置参数，词向量维度 ，默认为50维
        train_method:str
            word2vec设置参数，词向量训练方法，有两种（['cbow','skip']），默认使用cbow，
        iter: int
            word2vec设置参数，词向量训练迭代次数，默认为5
        """
        self.verbose = verbose
        self.size = size
        self.train_method = train_method
        self.iter = iter

        self.word2vec_model_root_path = os.path.dirname(__file__) + '/vector/'

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
        if self.verbose > 1:
            logging.debug('字典大小为：%d' % len(self.model.vocab))
            print '字典大小为：%d' % len(self.model.vocab)
            # -------------- print end : just print info -------------

    def save(self, output_path):
        self.model.save(output_path)

    def load(self, input_path):
        """
            加载w2v模型
            注意文件名(input_path)后缀，模型加载的方式会不同如果：
                - *.gem: 默认为使用 gensim.Word2Vec训练的模型 ，使用 Word2Vec.load(input_path) 方式加载
                - *.bin: 默认为使用 C版本的 word2vec模型训练，二进制，使用 Word2Vec.load_word2vec_format(input_path,binary=True) 方式加载

        """
        if input_path.endswith('.bin'):
            self.model = Word2Vec.load_word2vec_format(input_path, binary=True)

        elif input_path.endswith('.gem'):
            self.model = Word2Vec.load(input_path)
        else:
            raise NotImplementedError
        return self.model

    def get_word_similarity(self, word1, word2):
        """
            获取词的相似性，如果不在模型中，则返回0

        Notes
        ---------
            - 如果 word1 不在模型中,返回错误代码 L
            - 如果 word2 不在模型中,返回错误代码 R
            - 如果 word1 and word2 都不在模型中,返回错误代码 LR
            - 如果 word1 and word2 在模型中,返回 word1 跟 word2 的余弦相似性,范围为 [-1,1]

        Parameters
        ----------
        word1:str,unicode
        word2:str,unicode

        Returns
        -------
        float:
            相似性

        """

        assert self.model is not None, '请先 load() 模型！'

        flag = ''
        if word1 not in self.model.vocab:
            flag += 'L'
        if word2 not in self.model.vocab:
            flag += 'R'

        try:
            sim = self.model.similarity(word1, word2)
        except:
            # 如果都不在model中，返回错误代码
            sim = 0
        return sim,flag

    def transform_word2vec_model_name(self, flag):
        """
            根据 flag 转换成完整的 word2vec 模型文件名

        Parameters
        ----------
        flag: str
            版本标记
        Notes
        --------
        现在支持的有：
            - 50d_weibo_100w
            - 50d_weibo_1000w
            - 50d_sogou
            - 50d_v2.3Sa_word
            - 300d_weibo_100w
            - 400d_wiki_zh: 400d; vocab size:688025;
            - 300d_google_news

        Examples
        ----------
        >>> data_util = DataUtil()
        >>> data_util.transform_word2vec_model_name(flag='50d_weibo_100w')

        """

        if flag == '50d_weibo_100w':
            word2vec_model_file_path = self.word2vec_model_root_path + '50dim/vector1000000_50dim.gem'
        elif flag == '50d_weibo_1000w':
            word2vec_model_file_path = self.word2vec_model_root_path + '50dim/vector10000000_50dim.gem'
        elif flag == '50d_sogou':
            word2vec_model_file_path = self.word2vec_model_root_path + '50dim/08-12Sogou.gensim'
        elif flag == '50d_v2.3Sa_word':
            word2vec_model_file_path = self.word2vec_model_root_path + '50dim/v2.3_train_Sa_891_word_50dim.gem'
        elif flag == '300d_weibo_100w':
            word2vec_model_file_path = self.word2vec_model_root_path + '300dim/vector1000000_300dim.gem'
        elif flag == '400d_wiki_zh':
            word2vec_model_file_path = self.word2vec_model_root_path + '400dim/wiki.zh.text.model.gem'
        elif flag == '300d_google_news':
            word2vec_model_file_path = self.word2vec_model_root_path + '300dim/GoogleNews-vectors-negative300.bin'
        else:
            word2vec_model_file_path = self.word2vec_model_root_path + '50dim/vector1000000_50dim.gem'

        return word2vec_model_file_path

    def print_model_descibe(self):
        '''
            打印模型参数详情

        :return: 参数设置详情
        :rtype: dict 或 {}
        '''
        import pprint
        detail = {
            'verbose': self.verbose,
            'vector_size': self.model.layer1_size,
            'train_method': self.train_method,
            'iter': self.model.iter,
            'vocab size': len(self.model.vocab),
        }
        pprint.pprint(detail)
        logging.debug(detail)
        return detail


def test1():
    '''
        Test case： 训练word2vec

    '''

    # region 1、加载数据
    input_file1 = './sample_data/train_data_half_2090.csv'

    data = pd.read_csv(input_file1,
                       encoding='utf8',
                       sep='\t',
                       header=0)
    sentences = data['WORDS'].as_matrix()
    print '句子数：%d' % sentences.shape
    # endregion

    # region 2、训练
    util = Word2vecUtil(size=50,
                        train_method='skip'
                        )
    util.train(sentences)
    util.print_model_descibe()
    # endregion

    # region 3、测试
    most_similar_words = util.model.most_similar(u'喜欢')
    print ','.join([i for i, j in most_similar_words])
    util.save('vector/train_data_half_2090.gem')
    # endregion


def test2():
    input_file1 = './sample_data/v2.3_train_Sa_891.csv'

    data = pd.read_csv(input_file1,
                       encoding='utf8',
                       sep='\t',
                       index_col=0,
                       header=0)

    data = data[data['LABEL'] != u'其它#其它']
    data = data[data['LABEL'] != u'其它#捣乱']
    print(data.head())
    # 分词
    jieba_util = Jieba_Util()
    segment_sentence = lambda x: jieba_util.iter_each_word(
        sentence=x,
        sep=' ',
        need_segmented=True,
        full_mode=False,
        remove_stopword=False,
        replace_number=True,
        lowercase=True,
        zhs2zht=True,
        remove_url=True,
    )
    data['WORDS'] = data['SENTENCE'].apply(segment_sentence).as_matrix()
    sentences = data['WORDS'].as_matrix()
    print '句子数：%d' % sentences.shape
    # print(sentences[-1])
    # quit()
    util = Word2vecUtil(size=50,
                        train_method='cbow'
                        )
    util.train(sentences)
    util.print_model_descibe()

    most_similar_words = util.model.most_similar(u'机')
    most_similar_words = util.model.most_similar(u'喜')
    print ','.join([i for i, j in most_similar_words])
    util.save('vector/v2.3_train_Sa_891_word_50dim.gem')


def test_load_w2v():
    """
        Test Case: 加载w2v

    Returns
    -------

    """

    # # region 测试 google 新闻 词向量 ，英文，300dim
    # w2v_util = Word2vecUtil()
    # w2v_util.load(
    #     input_path=w2v_util.transform_word2vec_model_name(flag='300d_google_news'),
    # )
    # w2v_util.print_model_descibe()
    # # most_similar_words = w2v_util.model.most_similar(u'hello')
    # most_similar_words = w2v_util.model.most_similar(u'angry')
    # print ','.join([i for i, j in most_similar_words])
    # # endregion

    # region 测试 wiki_zh 词向量 ，300dim
    w2v_util = Word2vecUtil()
    w2v_util.load(
        input_path=w2v_util.transform_word2vec_model_name(flag='400d_wiki_zh'),
    )
    w2v_util.print_model_descibe()
    most_similar_words = w2v_util.model.most_similar(u'天气')
    print ','.join([i for i, j in most_similar_words])
    most_similar_words = w2v_util.model.most_similar(u'angry')
    print ','.join([i for i, j in most_similar_words])
    # endregion


def test_word_similarity():
    """
        Test Case： 词相似性

    Returns
    -------

    """
    w2v_util = Word2vecUtil()
    w2v_util.load(
        input_path=w2v_util.transform_word2vec_model_name(flag='400d_wiki_zh'),
    )
    print(w2v_util.get_word_similarity(u'发展', u'd进展'))


if __name__ == '__main__':
    # test1()
    # test2()
    # test_load_w2v()
    test_word_similarity()
