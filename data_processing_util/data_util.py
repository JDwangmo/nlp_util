# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-08-30'
    Email:   '383287471@qq.com'
    Describe: 一些经常用到的函数：
                1、transform_word2vec_model_name： 通过 名字获取 word2vec 模型名
                2、save_data: 保存数据成csv格式
                3、
"""

from __future__ import print_function
import pandas as pd

class DataUtil(object):
    def __init__(self):
        # 训练数据的根目录
        self.word2vec_model_root_path = '/home/jdwang/PycharmProjects/corprocessor/word2vec/vector/'
        self.jieba_util = None


    def transform_word2vec_model_name(self,flag):
        '''
            根据 flag 转换成完整的 word2vec 模型文件名

        :param flag:
        :return:
        '''

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
        else:
            word2vec_model_file_path = self.word2vec_model_root_path + '50dim/vector1000000_50dim.gem'

        return word2vec_model_file_path

    def save_data(self,data,path):
        '''
            保存DataFrame格式的数据

        :param data: 数据
        :param path: 数据文件的路径
        :return: None
        '''
        data.to_csv(path,
                    sep='\t',
                    header=True,
                    index=False,
                    encoding='utf8',
                    )



    def load_data(self,path):
        '''
            加载DataFrame格式的数据

        :param data: 数据
        :param path: 数据文件的路径
        :return: None
        '''
        data = pd.read_csv(path,
                           sep='\t',
                           header=0,
                           encoding='utf8',
                           index_col=0,
                           )
        return data


if __name__ == '__main__':
    data_util = DataUtil()
    # quit()

    data = pd.read_csv('/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/stable_vesion/v2.2/v2.2_train_Sa_893.csv',sep='\t')
    # data = data_util.load_data('/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/stable_vesion/20160708/v2.1_train_S_1786.csv')

    print(data.head())
    print(data.columns)
    print(data.shape)
    print(data[u'LABEL'].value_counts().sort_index())
    print(len(data[u'LABEL'].value_counts().sort_index()))
