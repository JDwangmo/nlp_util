# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-07-21'
    Email:   '383287471@qq.com'
    Describe: 数据集、语料库等交叉验证的常用方法
        1. transform_cv_data： 处理数据成交叉验证数据
"""
from __future__ import print_function
import numpy as np
import pandas as pd
import logging
import timeit

def data_split_k_fold(
        k=5,
        data=None,
        rand_seed = 2,
):
    '''
        将数据分为平均分为 K-部分，尽量按类别平均分

    :param k:
    :param data: (X,y)。 X 和 y 都是 array-like
    :type data: (array-like,array-like)
    :return:
    '''

    train_X_feature, train_y = data

    data = pd.DataFrame(data={'FEATURE_INDEX': range(len(train_X_feature)), 'LABEL': train_y})

    cross_validation_X_split = {i: [] for i in range(k)}
    cross_validation_y_split = {i: [] for i in range(k)}
    # 按各个类别来取数据
    rand = np.random.RandomState(rand_seed)
    for label, group in data.groupby(by=['LABEL'], axis=0):
        # print label
        # print group['SENTENCE_LENGTH'].value_counts()
        # 取出句子长度为length的所有句子
        start_index = rand.randint(0, k)
        sentences = rand.permutation(group['FEATURE_INDEX'].as_matrix())
        # print(sentences)

        for index, item in enumerate(sentences):
            # print index%k
            index = start_index + index
            cross_validation_X_split[index % k].append(train_X_feature[item])
            cross_validation_y_split[index % k].append(label)

    for x, y in zip(cross_validation_X_split.values(), cross_validation_y_split.values()):
        yield x, np.asarray(y,dtype=int)

def transform_cv_data(feature_encoder=None,
                      cv_data=None,
                      test_data=None,
                      word2vec_model_file_path=None,
                      **kwargs
                      ):
    '''
        将cv_data中的 k份数据创建 K份交叉验证的数据，每次取1份做测试，剩下做训练。
        另外将所有验证数据合并作为总体训练集，以test_data为测试数据
            1. 合并所有验证数据，作为总体训练
            2. 将所有训练数据和验证数据，测试数据全部转为特征编码

    :param feature_encoder: 特征编码器
    :param cv_data: array-like，[[],[]]，k份数据对应k个列表
    :param test_data: array-like，测试数据
    :param word2vec_model_file_path: 如果有提供，则返回init_weight
    :return: all_cv_data，包括训练数据（第一份，all_cv_data[0]）,all_cv_data[1:]为交叉验证数据集
    '''

    cv_x, cv_y = cv_data
    test_x, test_y = test_data
    test_y = np.asarray(test_y)

    # 合并验证数据为总体的训练数据
    train_x = np.concatenate(cv_x)
    train_y = np.concatenate(cv_y)
    # print(train_x)
    # pd.DataFrame(data={'SENTENCE':train_x,'LABEL':train_y}).to_csv('/home/jdwang/PycharmProjects/corprocessor/coprocessor/bow_model/bow_CNN_bow_WORD2VEC_oov_randomforest/result/train_%d.csv'%len(train_x),sep='\t',encoding='utf8',index=False)
    # pd.DataFrame(data={'SENTENCE':test_x,'LABEL':test_y}).to_csv('/home/jdwang/PycharmProjects/corprocessor/coprocessor/bow_model/bow_CNN_bow_WORD2VEC_oov_randomforest/result/test_%d.csv'%len(test_x),sep='\t',encoding='utf8',index=False)
    # quit()
    all_cv_data = []
    # 训练和测试数据
    x_features = feature_encoder.fit_transform(train_x)
    test_x_features = feature_encoder.transform(test_x)
    if kwargs['verbose']>0:
        print(','.join(feature_encoder.vocabulary))
        print('train shape:(%d,%d)'%(x_features.shape))
        print('test shape:(%d,%d)'%(test_x_features.shape))

    if kwargs.has_key('to_embedding_weight'):
        init_weight = feature_encoder.to_embedding_weight(word2vec_model_file_path)
        all_cv_data.append((x_features, train_y, test_x_features, test_y,init_weight))
    else:
        all_cv_data.append((x_features, train_y, test_x_features, test_y))



    for val_index in range(len(cv_x)):
        val_X, val_y = cv_x[val_index], cv_y[val_index]
        val_y= np.asarray(val_y)

        dev_X = cv_x[:val_index] + cv_x[val_index + 1:]
        dev_y = cv_y[:val_index] + cv_y[val_index + 1:]
        dev_X = np.concatenate(dev_X)
        dev_y = np.concatenate(dev_y)
        # 转为特征向量
        dev_X = feature_encoder.fit_transform(dev_X)
        val_X = feature_encoder.transform(val_X)
        if kwargs['verbose']>0:
            print(','.join(feature_encoder.vocabulary))
            print('dev_X shape:(%d,%d)'%dev_X.shape)
            print('val_X shape:(%d,%d)'%val_X.shape)

        if kwargs.has_key('to_embedding_weight'):
            init_weight = feature_encoder.to_embedding_weight(word2vec_model_file_path)
            all_cv_data.append((dev_X, dev_y, val_X, val_y, init_weight))
        else:
            all_cv_data.append((dev_X, dev_y, val_X, val_y))


    return all_cv_data
