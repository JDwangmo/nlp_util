# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-08-30'; 'last updated date: 2017-01-11'
    Email:   '383287471@qq.com'
    Describe: 一些经常用到的函数：
                1、transform_word2vec_model_name： 通过 名字获取 word2vec 模型名
                2、save_data: 保存数据成csv格式
                3、
"""

from __future__ import print_function
import pandas as pd


class DataUtil(object):
    __version__ = '1.1'

    def __init__(self):
        # 训练数据的根目录
        self.jieba_util = None

    def save_data(self, data, path):
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

    def load_data(self, path):
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

    data = pd.read_csv(
        '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/stable_vesion/v2.2/v2.2_train_Sa_893.csv',
        sep='\t')
    # data = data_util.load_data('/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/stable_vesion/20160708/v2.1_train_S_1786.csv')

    print(data.head())
    print(data.columns)
    print(data.shape)
    print(data[u'LABEL'].value_counts().sort_index())
    print(len(data[u'LABEL'].value_counts().sort_index()))
