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


class FeatureEncoder(object):
    '''
        Onehot特征编码器,将句子转成onehot编码(以字典索引形式表示,补齐),包含以下函数：
            1. segment_sentence：对句子分词
            2. build_dictionary：构建字典
            3. sentence_to_index：将原始字符串句子转为字典索引列表
            4. sentence_padding：将句子补齐
            5. build_encoder：构建编码器
            6. encoding_sentence：对句子编码
            7. get_sentence_length：对句子长度计算
            8. print_sentence_length_detail： 打印训练库句子详情.
            9. print_model_descibe: 打印模型的详情.
            10. sentence_index_to_onehot: 将索引转为onehot数据
            11. to_onehot_array： 生成训练库句子的onehot编码

        注意：
            1. 训练库中所有词，包括未知词字符（UNKOWN），的字典索引都是从1开始分配的，索引0是作为填充字符所用。
            2. 训练库字典大小 （train_data_dict_size）是不计入索引0的，只计算训练库中所有词和未知词字符（UNKOWN）。
    '''

    def __init__(self,
                 train_data=None,
                 need_segmented=True,
                 verbose=0,
                 full_mode=True,
                 remove_stopword=True,
                 replace_number=True,
                 lowercase = True,
                 zhs2zht = True,
                 remove_url = True,
                 sentence_padding_length=7,
                 padding_mode='center',
                 add_unkown_word=True,
                 mask_zero = True,
                 ):
        '''
            1. 初始化参数
            2. build feature encoder

            :param train_data: 训练句子列表:[[],[],...,[]]
            :type train_data: array-like.
            :param need_segmented: 数据处理选项,是否需要经过分词处理;如果为False,那么输入的数据不需要分词,提供的数据的每个句子的每个词要以空格分割.比如: ['我 要 买 手机','你好','早上 好'];如果为True,提供原始输入句子即可,比如:['我要买手机','你好','早上好'].
            :type need_segmented: bool
            :param verbose: 数值越大,输出越详细
            :type verbose: int
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
            :param add_unkown_word: 训练库字典的设置选项，是否在字典中增加一个未知词字符(UNKOWN)
            :type add_unkown_word: bool
            :param mask_zero: 训练库字典的设置选项，是否留出索引0，如果为True，表示0为掩码（空白符），不用做实际词的索引;若为False，则索引0作为普通词索引用。
            :type mask_zero: bool
            :param sentence_padding_length:  句子的补齐（截断）长度，默认为7
            :type sentence_padding_length: int
            :param padding_mode:  句子的补齐（截断）模式，有四种模式：
                                        1. center：如果小于sentence_padding_length的话往两边补0;如果超出sentence_padding_length的话，直接在后面截断。
                                        2. left：如果小于sentence_padding_length的话往左边补0;如果超出sentence_padding_length的话，直接在后面截断。
                                        3. right：如果小于sentence_padding_length的话往右边补0;如果超出sentence_padding_length的话，直接在后面截断。
                                        4. none：不补齐。
            :type padding_mode: str


        '''
        self.full_mode = full_mode
        self.remove_stopword = remove_stopword
        self.verbose = verbose
        self.train_data = train_data
        self.need_segmented = need_segmented
        self.replace_number = replace_number
        self.lowercase = lowercase
        self.zhs2zht = zhs2zht
        self.remove_url = remove_url
        self.add_unkown_word = add_unkown_word
        self.sentence_padding_length = sentence_padding_length
        self.mask_zero = mask_zero
        self.padding_mode = padding_mode

        # 检验参数合法性
        assert self.padding_mode in ['center','left','right','none'],'padding mode 只能取: center,left,right,none'


        # 初始化jieba分词器
        self.jieba_seg = Jieba_Util(verbose=self.verbose)
        # 切完词的句子
        self.segmented_sentences = None
        # 训练库提取出来的字典对象
        self.train_data_dict = None
        # 训练库提取出来的字典词汇列表
        self.vocabulary = None
        # 训练库提取出来的字典词汇个数
        self.train_data_dict_size = None
        # 训练库句子的字典索引形式
        self.train_index = None
        # 训练库句子的补齐的字典索引形式
        self.train_padding_index = None
        # 训练库句子装成onehot array
        self.onehot_array = None
        # word2vec 模型
        self.word2vec_model = None

        if verbose > 1:
            logging.debug('build feature encoder...')
            print 'build feature encoder...'
        self.build_encoder()

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

    def get_sentence_length(self,sentence):
        '''
            计算句子的长度，注意，这里的长度以词为单位，即分完词后统计。
                1. 对句子分词
                2. 对句子的词计算

        :param sentence: 句子
        :type sentence: str
        :return: 句子长度
        :rtype: int
        '''

        # 1. 分词
        segmented_senence = self.segment_sentence(sentence)
        # 2. 统计
        sentence_length = len(segmented_senence.split())

        return sentence_length


    def print_sentence_length_detail(self):
        '''
            打印训练库中句子的长度情况

        :return: 句子长度列表
        :rtype: list
        '''

        sentence_length = map(self.get_sentence_length,self.train_data)
        print '句子长度情况为：%s'%(str(sentence_length))
        print '句子最长长度为：%d'%(max(sentence_length))
        print '句子最短长度为：%d'%(min(sentence_length))
        print '句子平均长度为：%d'%(np.average(sentence_length))
        return sentence_length

    def get_unkown_vector(self, ndim=50):
        rand = np.random.RandomState(1337)
        return rand.rand(ndim)

    def get_w2vEmbedding(self, word):
        try:
            vector = self.word2vec_model[word]
        except:
            vector = self.get_unkown_vector(self.word2vec_model.vector_size)
        return np.asarray(vector)

    def to_embedding_weight(self,path):
        '''
            使用训练好的 word2vec 模型 将字典中每个词转为 word2vec向量，接着生成一个 Embedding层的初始权重形式，可用于初始化 Embedding 层的权重。
                1. 加载word2vec模型
                2.

        :param path: word2vec 模型文件路径
        :type path: str
        :return:
        '''
        self.word2vec_model = Word2Vec.load(path)

        # 若mask_zero=True，则需要为0留出一个位置，所有索引加1,embedding权重多加一行
        # 若mask_zero=False，则不需要为0留出一个位置，不用加1
        leave_for_zero = int(self.mask_zero)
        size = self.train_data_dict_size + leave_for_zero
        embedding_weights = np.zeros((size, self.word2vec_model.vector_size))
        for key,value in self.train_data_dict.token2id.items():
            embedding_weights[value+leave_for_zero,:] = self.get_w2vEmbedding(key)
        # todo 创建词向量字典
        self.embedding_weights = embedding_weights
        return embedding_weights

    def build_dictionary(self):
        '''
            1.对数据进行分词
            2.构建训练库字典,插入 一个特殊字符 'UNKOWN'表示未知词
        :return:
        '''

        # -------------- region start : 1.对数据进行分词 -------------
        if self.verbose > 1:
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
        if self.verbose > 1:
            logging.debug('-' * 20)
            print '-' * 20
        # -------------- region end : 1.对数据进行分词 ---------------
        # -------------- region start : 2.构建训练库字典 -------------
        if self.verbose > 1:
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
        if self.add_unkown_word:
            self.train_data_dict.add_documents([[u'UNKOWN']])

        self.train_data_dict_size = len(self.train_data_dict.keys())
        # 按索引从小到大排序
        self.vocabulary = [token for token,id in sorted(self.train_data_dict.token2id.items(),key=lambda x:x[1])]

        # -------------- print start : just print info -------------
        if self.verbose > 1:
            logging.debug('训练库字典为:%d' % (len(self.train_data_dict.keys())))
            print '训练库字典为:%d' % (len(self.train_data_dict.keys()))
            logging.debug(u'字典有:%s' % (','.join(self.train_data_dict.token2id.keys())))
            print u'字典有:%s' % (','.join(self.train_data_dict.token2id.keys()))
        # -------------- print end : just print info -------------

        # -------------- code start : 结束 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print '-' * 20
            # -------------- region end : 2.构建训练库字典 ---------------

    def sentence_to_index(self, sentence):
        """
            将 sentence 转换为 index,如果 token为OOV词,则分配为 UNKOWN

        :type sentence: str
        :param sentence: 以空格分割
        :return:
        """
        if self.add_unkown_word:
            unknow_token_index = self.train_data_dict.token2id[u'UNKOWN']
        else:
            unknow_token_index=0
        # 将训练库中所有句子的每个词映射到索引上,变成索引列表
        # 注意这里把所有索引都加1,目的是为了保留 索引0(用于补充句子),在神经网络上通过mask_zero忽略,实现变长输入
        if self.mask_zero:
            index = [self.train_data_dict.token2id.get(item, unknow_token_index) + 1 for item in sentence.split()]
        else:
            index = [self.train_data_dict.token2id.get(item, unknow_token_index) for item in sentence.split()]
        return index

    def sentence_padding(self, sentence):
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
            if self.padding_mode == 'center':
                sentence = np.concatenate((left_padding, sentence, right_padding), axis=0)
            elif self.padding_mode == 'left':
                sentence = np.concatenate((left_padding, right_padding, sentence), axis=0)
            elif self.padding_mode =='right':
                sentence = np.concatenate((sentence,left_padding, right_padding), axis=0)
            elif self.padding_mode=='none':
                sentence = sentence


        return sentence

    def word_index_to_onehot(self, index):
        '''
            注意:该方法跟[sentence_index_to_onehot]的区别。
            将词的索引转成 onehot 编码,比如：
                索引 1 -->[  0 , 0 , 0 , 0,  1]

        :param index: 一个词的字典索引
        :type index: int
        :return: onehot 编码，长度为 字典长度
        :rtype: np.array()
        '''
        # todo
        pass

        return None

    def sentence_index_to_onehot(self, index):
        '''
            注意:该方法跟[word_index_to_onehot]的区别。
            将句子的字典索引转成 onehot 编码比如：
                [1,2]-->[ 0 , 1 , 1 , 0 , 0 , 0 , 0 , 0 , 0,  0]

        :param index: 一个句子的字典索引
        :type index: list
        :return: onehot 编码，长度为 字典长度
        :rtype: np.array()
        '''

        onehot_array = np.zeros(self.train_data_dict_size+int(self.mask_zero),dtype=int)
        onehot_array[index] = 1

        return onehot_array


    def to_onehot_array(self):
        '''
            将所有训练库句子转成onehot编码的数组，保存在 self.onehot_array 中

        :return: onehot编码的数组
        '''
        self.onehot_array = np.asarray(map(self.sentence_index_to_onehot, self.train_index))
        return self.onehot_array

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
        if self.verbose > 1:
            logging.debug('-' * 20)
            print '-' * 20
            logging.debug('1.构建训练库字典')
            print '1.构建训练库字典'
        # -------------- code start : 开始 -------------

        # 构建训练库字典
        self.build_dictionary()

        # -------------- code start : 结束 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print '-' * 20
        # -------------- region end : 1.构建训练库字典 ---------------

        # -------------- region start : 2. 将训练句子转成字典索引形式 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print '-' * 20
            logging.debug('2. 将训练句子转成字典索引形式')
            print '2. 将训练句子转成字典索引形式'
        # -------------- code start : 开始 -------------

        # 将训练库中所有句子的每个词映射到索引上,变成索引列表
        self.train_index = map(self.sentence_to_index, self.segmented_sentences)
        # print train_index[:5]

        # -------------- code start : 结束 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print '-' * 20
        # -------------- region end : 2. 将训练句子转成字典索引形式 ---------------

        # -------------- region start : 3. 将句子补齐到等长 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print '-' * 20
            logging.debug('3. 将句子补齐到等长')
            print '3. 将句子补齐到等长'
        # -------------- code start : 开始 -------------

        # 将不等长的句子都对齐,超出padding_length长度的句子截断,小于的则补0
        train_padding_index = np.asarray(map(self.sentence_padding, self.train_index))
        self.train_padding_index = train_padding_index

        # -------------- code start : 结束 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print '-' * 20
            # -------------- region end : 3. 将句子补齐到等长 ---------------

    def encoding_sentence(self, sentence):
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

        sentence_index = self.sentence_to_index(seg_sentence)
        sentence_padding_index = self.sentence_padding(sentence_index)

        # -------------- code start : 结束 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print '-' * 20
        # -------------- region end : 2. 转为字典索引列表,之后补齐,输入为补齐的字典索引列表 ---------------

        return sentence_padding_index

    def print_model_descibe(self):
        '''
            打印模型参数详情

        :return: 参数设置详情
        :rtype: dict 或 {}
        '''
        import pprint
        detail = {'train_data_count': len(self.train_data),
                  'need_segmented': self.need_segmented,
                  'verbose': self.verbose,
                  'full_mode': self.full_mode,
                  'remove_stopword': self.remove_stopword,
                  'replace_number': self.replace_number,
                  'sentence_padding_length': self.sentence_padding_length,
                  'padding_mode': 'center',
                  'train_data_dict_size': self.train_data_dict_size,
                  'add_unkown_word': True,
                  'mask_zero': True,
                  }
        pprint.pprint(detail)
        logging.debug(detail)
        return detail

if __name__ == '__main__':
    train_data = ['你好，你好', '測試句子','无聊', '测试句子', '今天天气不错','买手机','你要买手机']
    test_data = '你好，你好,si'
    feature_encoder = FeatureEncoder(train_data=train_data,
                                     verbose=0,
                                     padding_mode='none',
                                     need_segmented=True,
                                     full_mode=True,
                                     remove_stopword=True,
                                     replace_number=True,
                                     lowercase=True,
                                     sentence_padding_length=7,
                                     add_unkown_word=True,
                                     mask_zero=True,
                                     zhs2zht=True,
                                     )
    embedding_weight = feature_encoder.to_embedding_weight('/home/jdwang/PycharmProjects/corprocessor/word2vec/vector/ood_sentence_vector1191_50dim.gem')
    print embedding_weight.shape
    # print embedding_weight
    print ','.join(feature_encoder.vocabulary)
    print feature_encoder.train_padding_index
    print feature_encoder.encoding_sentence(test_data)
    print feature_encoder.sentence_index_to_onehot(feature_encoder.encoding_sentence(test_data))
    quit()
    print feature_encoder.train_data_dict_size


    X = feature_encoder.to_onehot_array()
    print X

    # feature_encoder.print_sentence_length_detail()
    # feature_encoder.print_model_descibe()
