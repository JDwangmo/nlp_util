# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-07-21'
    Email:   '383287471@qq.com'
    Describe: 数据集、语料库等交叉验证的常用方法
        1. data_split_k_fold: 将数据分为平均分为 K-部分，尽量按类别平均分，直接获取数据
        2、get_splitted_k_fold_data_index： 将数据分为平均分为 K-部分，尽量按类别平均分，获取 数据 每份的标签索引
        3. get_k_fold_data: 获取 K折验证的训练和测试集（dev_set, val_set）
        4. transform_cv_data： 将 cv_data中的 k份数据 全部转为特征编码
        5. get_val_score： 获取某个参数设置下，模型的交叉验证情况
"""

from __future__ import print_function
import numpy as np
import pandas as pd
import logging
import copy

__all__ = ['data_split_k_fold',
           'get_k_fold_data',
           'transform_cv_data',
           'get_val_score',
           ]


def data_split_k_fold(
        k=5,
        data=None,
        rand_seed=2,
):
    '''
        将数据分为平均分为 K-部分，尽量按类别平均分

    :param k: k份
    :param data: (X,y)。 X 和 y 都是 array-like
    :type data: (array-like,array-like)
    :param rand_seed: 随机种子
    :type rand_seed: int
    :return:
    '''

    train_X_feature, train_y = data

    data = pd.DataFrame(data={'FEATURE_INDEX': range(len(train_X_feature)), 'LABEL': train_y})

    cross_validation_index_split = {i: [] for i in range(k)}
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
            index += start_index
            cross_validation_index_split[index % k].append(item)
            cross_validation_X_split[index % k].append(train_X_feature[item])
            cross_validation_y_split[index % k].append(label)

    for x, y in zip(cross_validation_X_split.values(), cross_validation_y_split.values()):
        yield x, np.asarray(y, dtype=int)


def get_splitted_k_fold_data_index(
        k=5,
        data=None,
        rand_seed=2,
):
    '''
        将数据分为平均分为 K-部分，尽量按类别平均分, 最终数据 每折数据的索引

    :param k: k份
    :param data: (X,y)。 X 和 y 都是 array-like
    :type data: (array-like,array-like)
    :param rand_seed: 随机种子
    :type rand_seed: int
    :return:
    '''

    train_X_feature, train_y = data
    # 用FEATURE_INDEX字段来记录数据的位置，以便后面复原数据
    data = pd.DataFrame(data={'FEATURE_INDEX': range(len(train_X_feature)), 'LABEL': train_y})

    cross_validation_index_split = {i: [] for i in range(k)}
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
            index += start_index
            cross_validation_index_split[index % k].append(item)

    for cv_index, data_index_list in cross_validation_index_split.iteritems():
        # print(cv_index,data_index_list)
        data.loc[data_index_list, 'CV_INDEX'] = cv_index

    data['CV_INDEX'] = data['CV_INDEX'].astype(dtype='int')
    # print(data['CV_INDEX'].values)
    return data['CV_INDEX'].values


def get_k_fold_data(
        k=3,
        train_data=None,
        test_data=None,
        include_train_data=True,
        rand_seed=0,
        **kwargs
):
    '''
        将数据分为K-fold,并获取交叉验证的  k份 (dev_set, val_set)
        每份 格式为 ：
            - train_or_dev_flag(0==train,1~=dev),
            - train_X,
            - train_y,
            - test_X,
            - test_y

    :param k: 几折
    :param train_data: (train_X, train_y)
    :type train_data: (array-like, array-like)
    :param test_data: (test_X, test_y)
    :type test_data: (array-like, array-like)
    :param include_train_data: cv_data中是否需要包含 (train_data,test_data)
    :type include_train_data: bool
    :return:
    '''

    train_X, train_y = train_data
    test_X, test_y = test_data
    cv_data = []
    if include_train_data:
        # 第一位为 标记位，0表示训练集
        cv_data.append([0, train_X, train_y, test_X, test_y])

    # 获取 K 份数据
    k_fold_data_x = []
    k_fold_data_y = []
    for x, y in data_split_k_fold(k=k, data=(train_X, train_y), rand_seed=rand_seed):
        k_fold_data_x.append(x)
        k_fold_data_y.append(y)

    for val_index in range(len(k_fold_data_x)):
        # val set
        val_X, val_y = k_fold_data_x[val_index], k_fold_data_y[val_index]
        val_y = np.asarray(val_y)
        # dev set
        dev_X = k_fold_data_x[:val_index] + k_fold_data_x[val_index + 1:]
        dev_y = k_fold_data_y[:val_index] + k_fold_data_y[val_index + 1:]
        dev_X = np.concatenate(dev_X)
        dev_y = np.concatenate(dev_y)
        # 第一位为 标记位，从1开始，表示验证集
        cv_data.append([val_index + 1, dev_X, dev_y, val_X, val_y])
    return cv_data


def transform_cv_data(
        feature_encoder=None,
        cv_data=None,
        **kwargs
):
    '''
        将 cv_data中的 k份数据 全部转为特征编码

    :param feature_encoder: 特征编码器
    :param cv_data: array-like，[[],[]]，k份数据对应k个列表
    :param kwargs: verbose[#,0], diff_train_val_feature_encoder[#,True]
    :return: cv_features,k 份，每一份 对应（dev_x_features, dev_y, val_x_features, val_y,feature_encoder）

    Notes
    ---------
    - 设置 diff_train_val_feature_encoder 为 0 ， feature encoder 都一样，不重置
    - 设置 diff_train_val_feature_encoder 为 1 ，可以保证 训练集上的feature encoder 和验证集上的 不同
    - 设置 diff_train_val_feature_encoder 为 2 ，可以保证 每次（含训练和验证上）的feature encoder 都 不同

    '''

    cv_features = []
    for flag, dev_x, dev_y, val_x, val_y in cv_data:
        # print(flag)
        dev_x_features = feature_encoder.fit_transform(dev_x, val_x)
        val_x_features = feature_encoder.transform(val_x)
        # feature_encoder.print_model_descibe()

        cv_features.append((flag, dev_x_features, dev_y, val_x_features, val_y, copy.deepcopy(feature_encoder)))
        if kwargs.get('verbose', 0) > 0:
            print(','.join(feature_encoder.vocabulary))
            print('vocabulary_size: %d' % (feature_encoder.vocabulary_size))
            print('dev shape:(%s)' % str(dev_x_features.shape))
            print('val shape:(%s)' % str(val_x_features.shape))

        if kwargs.get('verbose', 0) > 1:
            feature_encoder.print_sentence_length_detail(np.concatenate((dev_x, val_x), axis=0))
            feature_encoder.print_sentence_length_detail(dev_x)
            feature_encoder.print_sentence_length_detail(val_x)

        if kwargs.get('diff_train_val_feature_encoder', 0) == 0:
            # 不重置，都用同一个
            pass
        elif kwargs.get('diff_train_val_feature_encoder', 0) == 1:
            # 如果设置 让 训练集 和验证集 上的 feature encoder 不同，则训练完训练集feature encoder后清理对象数据
            if flag == 0:
                feature_encoder.reset()
        elif kwargs.get('diff_train_val_feature_encoder', 0) == 2:
            # 每次feature_encoder 都重置
            feature_encoder.reset()
        else:
            raise NotImplementedError

    # feature_encoder = None

    return cv_features


def get_val_score(
        estimator_class,
        cv_data,
        shuffle_data=False,
        **parameters
):
    """
        获取某个参数设置下，模型的交叉验证情况

    :param estimator_class: 分类器的类，必须实现了 get_model() 函数
    :param cv_data: 验证数据，第一份为 训练和测试数据，之后为验证数据
    :param shuffle_data: 是否打乱数据
    :param parameters: 参数, need_validation, get_cnn_middle_layer_output(#,False)
    :return: [test_accu] + [验证预测平均], train_acc, conv_middle_output_dev, conv_middle_output_val
    """

    # K折
    print('K折交叉验证开始...')
    # counter = 0
    test_acc = []
    train_acc = []
    conv_middle_output_dev = []
    conv_middle_output_val = []
    exclude_first = False
    while len(cv_data) != 0:
        flag, dev_X, dev_y, val_X, val_y, feature_encoder = cv_data.pop(0)
        # print(len(dev_X))
        print('-' * 80)
        if flag == 0:
            # 第一个数据是训练，之后是交叉验证
            print('训练:')
            # 因为第一份是训练排除掉
            exclude_first = True
        else:
            print('第%d个验证' % flag)
        parameters['dataset_flag'] = flag
        parameters['feature_encoder'] = feature_encoder
        # 构建分类器对象
        # print(parameters)
        estimator = estimator_class.get_model(**parameters)
        # estimator.print_model_descibe()
        if shuffle_data:
            dev_X = np.random.RandomState(0).permutation(dev_X)
            dev_y = np.random.RandomState(0).permutation(dev_y)
            # print(dev_y)

        # 拟合数据
        # dev_loss, dev_accuracy, val_loss, val_accuracy = 0,0,0,0
        dev_loss, dev_accuracy, val_loss, val_accuracy = estimator.fit((dev_X, dev_y), (val_X, val_y))


        if parameters.get('get_cnn_middle_layer_output', False):
            # 获取中间层输出
            conv_middle_output_dev = estimator.get_layer_output(dev_X)
            conv_middle_output_val = estimator.get_layer_output(val_X)

        print('dev:%f,%f' % (dev_loss, dev_accuracy))
        print('val:%f,%f' % (val_loss, val_accuracy))

        test_acc.append(val_accuracy)
        train_acc.append(dev_accuracy)
        if not parameters.get('need_validation', 'True'):
            break
            # counter += 1

    print('k折验证结果：%s' % test_acc)
    print('验证中训练数据结果：%s' % train_acc)
    print('验证中测试数据平均准确率：%f' % np.average(test_acc[int(exclude_first):]))
    print('测试结果汇总：%s' % (test_acc + [np.average(test_acc[int(exclude_first):])]))
    print('%s,%s' % (train_acc, test_acc))
    print('-' * 80)

    return test_acc + [np.average(test_acc[1:])], train_acc, conv_middle_output_dev, conv_middle_output_val
