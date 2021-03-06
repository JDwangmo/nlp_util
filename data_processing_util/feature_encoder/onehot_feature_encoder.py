# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-06-24'; 'last updated date: 2017-01-10'
    Email:   '383287471@qq.com'
    Describe: Onehot Encoder --- Onehot特征编码器,将句子转成 onehot编码
"""
from __future__ import print_function
import numpy as np
import logging
from data_processing_util.word2vec_util.word2vec_util import Word2vecUtil
from gensim.corpora.dictionary import Dictionary
from data_processing_util.jiebanlp.jieba_util import Jieba_Util

__version__ = '1.4'


class FeatureEncoder(object):
    """
        Onehot特征编码器,将句子转成 onehot编码
        函数列表为：
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
            12. reset： clear 数据

        注意：
            1. onehot编码 有两种形式：通过设置 to_onehot_array 切换
                - 字典索引形式表示,补齐 （默认是这种形式）
                - onehot 向量
            0. 训练库中所有词，包括未知词字符（UNKOWN），的字典索引都是从1开始分配的，索引0是作为填充字符所用。
            1. 训练库字典大小 （vocabulary_size）是计入索引0的，计算训练库中所有词和填充字符（PADDING）未知词字符（UNKOWN），如果不使用可以关闭。
    """
    __version__ = '1.4'

    def __init__(self,
                 need_segmented=True,
                 verbose=0,
                 full_mode=True,
                 feature_type='seg',
                 remove_stopword=True,
                 replace_number=True,
                 lowercase=True,
                 zhs2zht=True,
                 remove_url=True,
                 sentence_padding_length=7,
                 padding_mode='center',
                 add_unkown_word=True,
                 to_onehot_array=False,
                 word2vec_to_solve_oov=False,
                 **kwargs
                 ):
        """
            Onehot特征编码器,将句子转成 onehot 编码(以字典索引形式表示,补齐)
            1. 初始化参数
            2. build feature encoder

            :param need_segmented: 数据处理选项,是否需要经过分词处理;如果为False,那么输入的数据不需要分词,提供的数据的每个句子的每个词要以空格分割.比如: ['我 要 买 手机','你好','早上 好'];如果为True,提供原始输入句子即可,比如:['我要买手机','你好','早上好'].
            :type need_segmented: bool
            :param verbose: 数值越大,输出越详细
            :type verbose: int
            :param full_mode: jieba分词选项,是否使用 full mode,默认为True
            :type full_mode: bool
            :param feature_type: 模型设置选项,选择不同粒度的特征单位， 目前只支持 word,seg和 word_seg。
                - word：直接以字为单位，比如 我要买手机--->我 要 买 手 机
                - seg：分词后的词单元为单位，比如 我要买手机--->我 要 买 手机
                - word_seg：分词后的字和词为单位，去重，比如 我要买手机--->我 要 买 手机 手 机
                - word_seg_concat：分词后的字和词为单位，不去重，比如 我要买手机--->我 要 买 手 机 我 要 买 手机
            :type feature_type: str
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
            :param sentence_padding_length:  句子的补齐（截断）长度，默认为7
            :type sentence_padding_length: int
            :param padding_mode:  句子的补齐（截断）模式，有四种模式：
                                        1. center：如果小于sentence_padding_length的话往两边补0;如果超出sentence_padding_length的话，直接在后面截断。
                                        2. left：如果小于sentence_padding_length的话往左边补0;如果超出sentence_padding_length的话，直接在后面截断。
                                        3. right：如果小于sentence_padding_length的话往右边补0;如果超出sentence_padding_length的话，直接在后面截断。
                                        4. none：不补齐。
            :type padding_mode: str
            :param to_onehot_array: 输出 onehot array，还是字典索引 array，默认为False，输出字典索引,
            :type to_onehot_array: bool
            :param word2vec_to_solve_oov: 使用word2vec扩展oov词
            :type word2vec_to_solve_oov: bool
            :param kwargs:
                - word2vec_model_file_path:
                - vocabulary_including_test_set: (default,True)
                - update_dictionary: (default,True)
                - 等

        """
        self.full_mode = full_mode
        self.feature_type = feature_type
        self.remove_stopword = remove_stopword
        self.verbose = verbose
        self.need_segmented = need_segmented
        self.replace_number = replace_number
        self.lowercase = lowercase
        self.zhs2zht = zhs2zht
        self.remove_url = remove_url
        self.add_unkown_word = add_unkown_word
        self.sentence_padding_length = sentence_padding_length
        self.padding_mode = padding_mode
        self.to_onehot_array = to_onehot_array
        self.word2vec_to_solve_oov = word2vec_to_solve_oov
        self.kwargs = kwargs

        # 检验参数合法性
        assert self.padding_mode in ['center', 'left', 'right', 'none'], 'padding mode 只能取: center,left,right,none'
        assert self.feature_type in ['word', 'seg', 'word_seg',
                                     'word_seg_concat'], 'feature type 只能取: word,seg和word_seg'

        # 初始化jieba分词器
        if need_segmented:
            self.jieba_seg = Jieba_Util(verbose=self.verbose)
        # 训练库提取出来的字典对象
        self.train_data_dict = None
        # 训练库提取出来的字典词汇列表
        self.vocabulary = None
        # 训练库提取出来的字典词汇个数
        self.vocabulary_size = None
        # UNKOWN字符的索引
        self.unknow_token_index = None
        # PADDING字符的索引
        self.padding_token_index = None

        # region NOTE: 这些变量不再维护，因为消耗内存
        # 原始训练数据
        # self.train_data = None
        # 切完词的句子
        # self.segmented_sentences = None
        # 训练库句子的字典索引形式
        # self.train_index = None
        # 训练库句子的补齐的字典索引形式
        # self.train_padding_index = None
        # 训练库句子装成onehot array
        # endregion
        self.train_onehot_array = None
        # word2vec 模型
        self.word2vec_model = None
        if word2vec_to_solve_oov:
            assert kwargs.has_key('word2vec_model_file_path'), '请提供 属性 word2vec_model_file_path'
            # 加载word2vec模型
            w2v_util = Word2vecUtil()
            self.word2vec_model = w2v_util.load(kwargs.get('word2vec_model_file_path'))

        if verbose > 1:
            logging.debug('build feature encoder...')
            print('build feature encoder...')
            # self.fit_transform()

    def segment_sentence(self, sentence):
        """
        对句子进行分词,使用jieba分词

        :param sentence: 句子
        :type sentence: str
        :return: 分完词句子，以空格连接
        :rtype: str
        """

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
        elif self.feature_type == 'word_seg_concat':
            # 先字后词拼接，不去重
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
            segmented_sentence = ' '.join(word + seg)
        else:
            assert False, '不支持其他粒度的切分！'

        return segmented_sentence

    def get_sentence_length(self, sentence):
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

    def print_sentence_length_detail(
            self,
            data=None,
            lengths=[7, 10, 15, 20,50,80,100],
    ):
        """
            打印训练库中句子的长度情况

        :type lengths: list
        :param lengths: 长度界限列表
        :return: 句子长度列表
        :rtype: list
        """
        if self.need_segmented:
            sentence_length = map(self.get_sentence_length, data)
        else:
            sentence_length = map(lambda x: len(x.split()), data)
        for l in lengths:
            le_this_len = sum(np.asarray(sentence_length) <= l) / (1.0 * len(sentence_length))
            print('句子长度小于等于%d的有：%f' % (l, le_this_len))

        print('句子长度情况为：%s' % (str(sentence_length)))
        print('句子最长长度为：%d' % (max(sentence_length)))
        print('句子最短长度为：%d' % (min(sentence_length)))
        print('句子平均长度为：%d' % (np.average(sentence_length)))
        return sentence_length

    def get_unkown_vector(self, ndim=50):
        rand = np.random.RandomState(1337)
        return rand.uniform(-0.25, 0.25, ndim)

    def get_w2vEmbedding(self, word):
        """
            返回词向量

        Returns
        -------
        (array,str)
        """

        try:
            if word == u'PADDING':
                vector = np.zeros(self.word2vec_model.vector_size)
                flag = 'PADDING'
            elif word == u'UNKOWN':
                # 当训练
                vector = self.get_unkown_vector(self.word2vec_model.vector_size)
                flag = 'NO_IN_MODEL_VOCAB'
            else:
                vector = self.word2vec_model[word]
                flag = 'OK'
        except:
            vector = self.get_unkown_vector(self.word2vec_model.vector_size)
            if self.verbose > 1:
                print('OOV: %s' % word)
            flag = 'NO_IN_W2V'
        return np.asarray(vector), flag

    def to_embedding_weight(self, path):
        """
            使用训练好的 word2vec 模型 将字典中每个词转为 word2vec向量，接着生成一个 Embedding层的初始权重形式，可用于初始化 Embedding 层的权重。
                1. 加载word2vec模型
                2.

        :param path: word2vec 模型文件路径
        :type path: str
        :return:
        """

        if self.word2vec_model is None:
            w2v_util = Word2vecUtil()
            self.word2vec_model = w2v_util.load(path)
        size = self.vocabulary_size

        embedding_weights = np.zeros((size, self.word2vec_model.vector_size))
        words_count_no_in_w2v = 0
        words_count_no_in_vacab = 0
        words_count_in = 0
        words_count_paddding = 0
        for key, value in self.train_data_dict.token2id.items():
            vector, flag = self.get_w2vEmbedding(key)
            embedding_weights[value, :] = vector
            if flag == 'NO_IN_W2V':
                words_count_no_in_w2v += 1
            if flag == 'NO_IN_MODEL_VOCAB':
                words_count_no_in_vacab += 1
            if flag == 'OK':
                words_count_in += 1
                # print(key)
            if flag == 'PADDING':
                words_count_paddding += 1
        if self.verbose > 0:
            print('没有出现在w2v模型中的词有：%d个' % (words_count_no_in_w2v))
            print('没有出现在模型vocab中的词有：%d个' % (words_count_no_in_vacab))
            print('出现在w2v模型中的词有：%d个' % (words_count_in))

        # self.embedding_weights = embedding_weights

        return embedding_weights

    def build_dictionary(self, train_X=None, test_X=None):
        """
            1.对数据进行分词
            2.构建训练库字典,插入 一个特殊字符 'UNKOWN'表示未知词

        Parameters
        ----------
        train_X : array-like
        test_X : array-like

        Returns
        --------
        object:
            self
        """

        # region -------------- 1.将训练集和测试集合并 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('1.将训练集和测试集合并')
            print('1.将训练集和测试集合并')
        if self.kwargs.get('vocabulary_including_test_set', True):
            X = np.concatenate((train_X, test_X), axis=0)
        else:
            X = train_X

        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
        # endregion -------------- 1.将训练集和测试集合并 ---------------

        # region -------------- 2.对数据进行分词 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('对数据进行分词')
            print('对数据进行分词')
        # -------------- code start : 开始 -------------
        if self.need_segmented:
            segmented_sentences = map(self.segment_sentence, X)
        else:
            segmented_sentences = X
        # -------------- code start : 结束 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
        # endregion -------------- 2.对数据进行分词 ---------------

        # region -------------- 3. 将句子补齐到等长 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('2. 将句子补齐到等长')
            print('2. 将句子补齐到等长')
        # -------------- code start : 开始 -------------

        # 将不等长的句子都对齐,超出padding_length长度的句子截断,小于的则补 PADDING
        padded_sentences = np.asarray(map(self.sentence_padding, segmented_sentences))

        # endregion -------------- 3. 将句子补齐到等长 -------------

        # region -------------- region start : 4.构建训练库字典 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('4.构建训练库字典')
            print('4.构建训练库字典')
        # -------------- code start : 开始 -------------
        logging.debug('=' * 20)
        logging.debug('首先,构建训练库字典,然后将每个词映射到一个索引,再将所有句子映射成索引的列表')

        # 将训练库所有句子切分成列表,构成 2D的训练文档,每个单元是一个token,
        # 比如: [['今年','你','多少岁'],['你', '二十四','小时','在线','吗'],...]
        # 将分完词句子转成合适的数据格式
        train_document = map(lambda x: x.split(), padded_sentences)
        # 获取训练库字典
        if self.padding_mode != 'none':
            # 为了确保padding的索引是0,所以在最前面加入 PADDING
            train_document.insert(0, [u'PADDING'])
        self.train_data_dict = Dictionary.from_documents(train_document)

        # 更新字典,再字典中添加特殊符号,其中
        # UNKOWN表示未知字符,即OOV词汇
        if self.add_unkown_word:
            self.train_data_dict.add_documents([[u'UNKOWN']])

        # 获取padding和UNKOWN 的字典索引
        self.padding_token_index = self.train_data_dict.token2id.get(u'PADDING', -1)
        self.unknow_token_index = self.train_data_dict.token2id.get(u'UNKOWN', -1)

        self.vocabulary_size = len(self.train_data_dict.keys())
        # 按索引从小到大排序
        self.vocabulary = [token for token, id in sorted(self.train_data_dict.token2id.items(), key=lambda x: x[1])]
        # print(self.vocabulary_size)
        # print((self.train_data_dict.token2id.items()))
        # quit()

        # -------------- print start : just print info -------------
        if self.verbose > 1:
            logging.debug('训练库字典为:%d' % (len(self.train_data_dict.keys())))
            print('训练库字典为:%d' % (len(self.train_data_dict.keys())))
            logging.debug(u'字典有:%s' % (','.join(self.train_data_dict.token2id.keys())))
            print(u'字典有:%s' % (','.join(self.train_data_dict.token2id.keys())))
        # -------------- print end : just print info -------------

        # -------------- code start : 结束 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)

        # endregion -------------- 4.构建训练库字典 ---------------

        return padded_sentences

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

    def word_similarity(self, word2vec_model, word1, word2):
        '''
        计算两个词的相似性
        :param word1:
        :param word2:
        :return:
        '''
        try:
            return word2vec_model.n_similarity(word1, word2)
        except:
            return 0

    def sentence_to_index(self, sentence):
        """
            将 sentence 转换为 index,如果 token为OOV词,则分配为 UNKOWN


        Parameters
        ----------
        sentence: str
            以空格分割

        """

        if self.add_unkown_word:
            unknow_token_index = self.train_data_dict.token2id[u'UNKOWN']
        else:
            unknow_token_index = 0
        # 将训练库中所有句子的每个词映射到索引上,变成索引列表
        index = [self.train_data_dict.token2id.get(item, unknow_token_index) for item in sentence.split()]
        if self.verbose > 1:
            if index.__contains__(unknow_token_index):
                print('unknow_token_index:%d' % unknow_token_index)
                print('出现字典OOV')
                print(sentence)
                print(index)
        # assert not index.__contains__(-1),u'出现OOV词'
        return index

    def sentence_padding(self, sentence):
        '''
            将不等长的句子都对齐,超出padding_length长度的句子截断,小于的则补 PADDING

        :type sentence: str
        :param sentence: 句子,词之间以 空格 分割
        :return: 返回补齐后的句子，以空格分割
        :type: str
        '''

        padding_length = self.sentence_padding_length
        # print(sentence)
        sentence = sentence.split()
        sentence_length = len(sentence)
        # print(sentence_length)
        if sentence_length > padding_length:
            # logging.debug(u'对句子进行截断:%s' % (sentence))

            sentence = sentence[:padding_length]

            # logging.debug(u'对句子进行截断后:%s' % (' '.join(seg[:padding_length])))
            # print(u'对句子进行截断后:%s' % (' '.join(seg[:padding_length])))
        elif sentence_length < padding_length:
            should_padding_length = padding_length - sentence_length
            left_padding = np.asarray(['PADDING'] * (should_padding_length / 2))
            right_padding = np.asarray(['PADDING'] * (should_padding_length - len(left_padding)))
            if self.padding_mode == 'center':
                sentence = np.concatenate((left_padding, sentence, right_padding), axis=0)
            elif self.padding_mode == 'left':
                sentence = np.concatenate((left_padding, right_padding, sentence), axis=0)
            elif self.padding_mode == 'right':
                sentence = np.concatenate((sentence, left_padding, right_padding), axis=0)
            elif self.padding_mode == 'none':
                sentence = sentence
            else:
                raise NotImplemented

        sentence = ' '.join(sentence)
        return sentence

    def sentence_index_to_onehot(self, index):
        '''
            注意:该方法跟[sentence_index_to_bow]的区别。
            将词的索引转成 onehot 编码,比如：
                索引 1 -->[  0 , 0 , 0 , 0,  1]

        :param index: 一个词的字典索引
        :type index: list
        :return: onehot 编码，shape为 (句子长度，字典长度)
        :rtype: np.array()
        '''

        onehot_array = []

        for item in index:
            temp = np.zeros(self.vocabulary_size, dtype=int)
            if item == 0:
                pass
            else:
                temp[item - 1] = 1

            onehot_array.append(temp)

        # onehot_array = np.concatenate(onehot_array,axis=1)
        onehot_array = np.asarray(onehot_array)
        return onehot_array

    def sentence_index_to_bow(self, index):
        '''
            注意:该方法跟[word_index_to_onehot]的区别。
            将句子的字典索引转成 词包向量 编码比如：
                [1,2]-->[ 0 , 1 , 1 , 0 , 0 , 0 , 0 , 0 , 0,  0]

        :param index: 一个句子的字典索引
        :type index: list
        :return: bow 编码，长度为 字典长度
        :rtype: np.array()
        '''

        onehot_array = np.zeros(self.vocabulary_size, dtype=int)

        onehot_array[index] = 1

        return onehot_array

    def batch_sentence_index_to_onehot_array(self, sentence_indexs):
        '''
            将所有训练库句子转成onehot编码的数组，保存在 self.onehot_array 中

        :return: onehot编码的数组
        '''

        self.onehot_array = np.asarray(map(self.sentence_index_to_onehot, sentence_indexs))
        return self.onehot_array

    def fit_transform(self,
                      train_data=None,
                      test_data=None):
        return self.fit(train_data, test_data).transform(train_data)

    def fit(self,
            train_X=None,
            test_X=None
            ):
        """
            build feature encoder ---- 构建训练库字典

            Notes
            ------
            update_dictionary: 设置 再次调用fit()函数时，是否更新字典，默认为 True，即只在第一次调用fit()函数时才更新 字典
            vocabulary_including_test_set： 设置 是否 字典是否包含测试集的词汇，默认包含，即字典包含训练集和测试集的所有词汇。
                - 设置为 False ，则 字典只包含训练集中的词汇

        Parameters
        ----------
        train_X: array-like
            训练句子列表:['','',...,'']
        test_X: array-like
            测试句子列表:['','',...,'']

        Returns
        -------
        object:
            编码后的列表
        """

        if not self.kwargs.get('update_dictionary', True):
            # 假如不更新字典，则如果原有的字典在，就直接用原有的字典即可
            if self.vocabulary is not None:
                return self

        logging.debug('=' * 20)
        if train_X is None:
            logging.debug('没有输入训练数据!')
            assert False, '没有输入训练数据!'

        if self.kwargs.get('vocabulary_including_test_set', True):
            if test_X is None:
                logging.debug('vocabulary_including_test_set=True，构建字典需要全部数据，请输入测试数据!')
                assert False, 'vocabulary_including_test_set=True，构建字典需要全部数据，请输入测试数据!'

        # region -------------- 1.构建训练库字典 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('1.构建训练库字典')
            print('1.构建训练库字典')
        # -------------- code start : 开始 -------------

        # 构建训练库字典
        self.build_dictionary(train_X, test_X)

        # -------------- code start : 结束 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
        # endregion -------------- 1.构建训练库字典 ---------------

        return self

    def transform_sentence(self, sentence):
        """
            转换一个句子的格式。跟训练数据一样的操作,对输入句子进行 padding index 编码,将sentence转为补齐的字典索引
                1. 分词
                2. 转为字典索引列表,之后补齐,输入为补齐的字典索引列表
                    - 当 参数 to_onehot_array = True （默认为 False）时，直接返回 字典索引 ;
                    - 当 参数 to_onehot_array = False （默认为 False）时，进入第3步，进一步转换成 onehot 向量 ;
                3. 每个词的字典索引变成onehot向量
                    - 这一步不一定会执行
                    - to_onehot_array = True 时， 执行

        :param sentence: 输入句子,不用分词,进来后会有分词处理
        :type sentence: str
        :return: 补齐的字典索引
        :rtype: array-like
        """

        assert self.train_data_dict is not None, '请先fit_transform()模型'

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
        # endregion -------------- region end : 1. 分词 ---------------

        # region -------------- 2. 转为字典索引列表,之后补齐,输出为补齐的字典索引列表 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('2. 转为字典索引列表,之后补齐,输出为补齐的字典索引列表')
            print('2. 转为字典索引列表,之后补齐,输入为补齐的字典索引列表')
        # -------------- code start : 开始 -------------

        paded_sentence = self.sentence_padding(seg_sentence)
        sentence_index = self.sentence_to_index(paded_sentence)

        # -------------- code start : 结束 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
        # endregion -------------- region end : 2. 转为字典索引列表,之后补齐,输出为补齐的字典索引列表 ---------------

        # region -------------- 3. 将每个词的字典索引变成onehot向量 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('3. 将每个词的字典索引变成onehot向量')
            print('3. 将每个词的字典索引变成onehot向量')
        # -------------- code start : 开始 -------------

        if self.to_onehot_array:
            onehot_array = self.sentence_index_to_onehot(sentence_index)
        else:
            onehot_array = sentence_index

        # -------------- code start : 结束 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
        # endregion -------------- region end : 3. 将每个词的字典索引变成onehot向量 ---------------

        return onehot_array

    def transform(self, X):
        '''
            批量转换数据，跟训练数据一样的操作,对输入句子进行 padding index 编码,将sentence转为补齐的字典索引
                1. 直接调用 self.transform_sentence 进行处理

        :param sentence: 输入句子
        :type sentence: array-like
        :return: 补齐的字典索引
        :rtype: array-like
        '''

        index = map(lambda x: self.transform_sentence(x), X)
        # print train_index[:5]

        return np.asarray(index)

    def reset(self):
        """
        清理对象中的数据
           - self.vocabulary

        """
        self.vocabulary = None

    def print_model_descibe(self):
        '''
            打印模型参数详情

        :return: 参数设置详情
        :rtype: dict 或 {}
        '''
        import pprint
        detail = {'train_data_count': len(self.train_data),
                  'need_segmented': self.need_segmented,
                  'feature_type': self.feature_type,
                  'verbose': self.verbose,
                  'full_mode': self.full_mode,
                  'remove_stopword': self.remove_stopword,
                  'replace_number': self.replace_number,
                  'sentence_padding_length': self.sentence_padding_length,
                  'padding_mode': 'center',
                  'vocabulary_size': self.vocabulary_size,
                  'padding_token_index': self.padding_token_index,
                  'unknow_token_index': self.unknow_token_index,
                  'add_unkown_word': True,
                  'mask_zero': True,
                  }
        pprint.pprint(detail)
        logging.debug(detail)
        return detail


def test_word_onehot():
    '''
        测试基于 中文字 或 英文单词 的 onehot 编码

    :return:
    '''

    feature_encoder = FeatureEncoder(
        verbose=0,
        # 设置padding mode
        padding_mode='center',
        need_segmented=True,
        # full_mode 选择 False
        full_mode=False,
        # feature_type 选择 word
        feature_type='word_seg_concat',
        remove_stopword=True,
        replace_number=True,
        remove_url=True,
        lowercase=True,
        # 设置补齐长度
        sentence_padding_length=5,
        add_unkown_word=True,
        zhs2zht=True,
        # 输出 onehot array，还是字典索引 array
        to_onehot_array=False,
        word2vec_to_solve_oov=False,

    )
    train_padding_index = feature_encoder.fit_transform(train_data=train_data, test_data=train_data)

    embedding_weight = feature_encoder.to_embedding_weight(
        feature_encoder,
        '/home/jdwang/PycharmProjects/nlp_util/data_processing_util/word2vec_util/vector/v2.3_train_Sa_891_word_50dim.gem')

    quit()
    # 拟合数据
    print(feature_encoder.print_model_descibe())
    print(','.join(feature_encoder.vocabulary))

    for item in feature_encoder.padded_sentences:
        print(item)

    print(train_padding_index)
    test_padding_index = feature_encoder.transform(test_data)
    print(test_padding_index)
    quit()
    embedding_weight = feature_encoder.to_embedding_weight(
        '/home/jdwang/PycharmProjects/corprocessor/word2vec/vector/50dim/ood_sentence_vector1191_50dim.gem')
    print(embedding_weight.shape)
    # print(embedding_weight)
    print(feature_encoder.transform_sentence(test_data[0]))

    print(feature_encoder.sentence_index_to_bow(feature_encoder.transform_sentence(test_data[0])))

    quit()

    print(X)

    # feature_encoder.print_sentence_length_detail()
    # feature_encoder.print_model_descibe()


if __name__ == '__main__':
    train_data = ['妈B', 'ABCD', 'ch2r你好', '你好，你好', '測試句子', '无聊', '测试句子', '今天天气不错', '买手机', '你要买手机']
    test_data = ['您好', '今天不错', '手机卖吗']
    test_word_onehot()
