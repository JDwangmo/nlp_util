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
import copy
__all__ = ['data_split_k_fold',
           'transform_cv_data',
           'get_val_score'
           ]


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
import pickle
def load_val(path):
    with open(path,'rb') as train_file:
        X = pickle.load(train_file)
        y = pickle.load(train_file)
        val_X = pickle.load(train_file)
        val_y = pickle.load(train_file)
        return (X,y),(val_X,val_y)

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
        print('train shape:(%s)'%str(x_features.shape))
        print('test shape:(%s)'%str(test_x_features.shape))

    if kwargs.has_key('to_embedding_weight'):
        init_weight = feature_encoder.to_embedding_weight(word2vec_model_file_path)
        all_cv_data.append((x_features, train_y, test_x_features, test_y,copy.copy(feature_encoder),init_weight))
    else:
        all_cv_data.append((x_features, train_y, test_x_features, test_y,copy.copy(feature_encoder),None))

    # print(feature_encoder.vocabulary_size)
    # print(x_features.shape)
    # quit()

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
            print('dev_X shape:(%s)'%str(dev_X.shape))
            print('val_X shape:(%s)'%str(val_X.shape))

        if kwargs.has_key('to_embedding_weight'):
            init_weight = feature_encoder.to_embedding_weight(word2vec_model_file_path)
            all_cv_data.append((dev_X, dev_y, val_X, val_y, copy.copy(feature_encoder),init_weight))
        else:
            all_cv_data.append((dev_X, dev_y, val_X, val_y,copy.copy(feature_encoder),None))


    return all_cv_data


def get_val_score(estimator_class, cv_data, fout, **parameters):
    '''
        获取某个参数设置下的交叉验证情况

    :param estimator_class: 分类器的类，必须实现了 fit 函数
    :param cv_data: 验证数据，第一份为 训练和测试数据，之后为验证数据
    :param fout: 文件的输出
    :param parameters: 参数
    :return: [test_accu] + [验证预测平均], train_acc
    '''

    # K折
    print('K折交叉验证开始...')
    counter = 0
    test_acc = []
    train_acc = []
    for dev_X, dev_y, val_X, val_y, feature_encoder,init_weight in cv_data:
        # print(len(dev_X))
        print('-' * 80)
        fout.write('-' * 80 + '\n')
        if counter == 0:
            # 第一个数据是训练，之后是交叉验证
            print('训练:')
            fout.write('训练\n')

        else:
            print('第%d个验证' % counter)
            fout.write('第%d个验证\n' % counter)
        parameters['feature_encoder'] = feature_encoder
        parameters['embedding_init_weight'] = init_weight
        # 构建分类器对象
        estimator = estimator_class(**parameters)
        # estimator.print_model_descibe()
        # 拟合数据
        dev_loss, dev_accuracy, val_loss, val_accuracy = estimator.fit((dev_X, dev_y), (val_X, val_y))

        print('dev:%f,%f' % (dev_loss, dev_accuracy))
        print('val:%f,%f' % (val_loss, val_accuracy))
        fout.write('dev:%f,%f\n' % (dev_loss, dev_accuracy))
        fout.write('val:%f,%f\n' % (val_loss, val_accuracy))

        train_X_conv2_output = estimator.get_layer_output(dev_X,layer='conv1', transform_input=False)
        test_X_conv2_output = estimator.get_layer_output(val_X, layer='conv1', transform_input=False)
        from traditional_classify.bow_rf.bow_rf_model import BowRandomForest
        bow_rf = BowRandomForest(
            # rand_seed=seed,
            verbose=0,
            n_estimators=2000,
            min_samples_leaf=1,
            feature_encoder=None,
        )
        # (bow6_train_X, bow6_y), (bow6_val_X, bow6_val_y) = load_val(
        #     '/home/jdwang/PycharmProjects/corprocessor/coprocessor/bow_model/bow_WORD2VEC_oov_randomforest/TrainSet_region7_conv&hidden_convFeature160_d1.pickle')
        # (bowL_train_X, bowL_y), (bowL_val_X, bowL_val_y) = load_val(
        #     '/home/jdwang/PycharmProjects/corprocessor/coprocessor/bow_model/bow_WORD2VEC_oov_randomforest/TrainSet_bowl_conv&hidden_convfeature150_d1.pickle')
        # cv_x = []
        # cv_y = []
        # for x, y in data_split_k_fold(k=3, data=(bow6_train_X, bow6_y), rand_seed=3):
        #     cv_x.append(x)
        #     cv_y.append(y)
        #
        # # quit()
        # print(val_y)
        # print(cv_y[0])

        # t6_x = np.concatenate((cv_x[1],cv_x[2]))
        # te6_x = cv_x[0]
        # cv_x = []
        # cv_y = []
        # for x, y in data_split_k_fold(k=3, data=(bowL_train_X, bowL_y), rand_seed=3):
        #     cv_x.append(x)
        #     cv_y.append(y)
        #
        # # quit()
        # print(val_y)
        # print(cv_y[0])
        #
        # tl_x = np.concatenate((cv_x[1],cv_x[2]))
        # tel_x = cv_x[0]

        # train_X_conv2_output = np.concatenate((train_X_conv2_output,t6_x,tl_x),axis=1)
        # train_X_conv2_output = tl_x
        # test_X_conv2_output = np.concatenate((test_X_conv2_output,te6_x,tel_x),axis=1)
        # test_X_conv2_output = tel_x
        # from sklearn.preprocessing import Normalizer
        # normalizer = Normalizer(norm='l2')
        # train_X_conv2_output = normalizer.fit_transform(train_X_conv2_output)
        # test_X_conv2_output = normalizer.transform(test_X_conv2_output)

        bow_rf.fit(train_data=(train_X_conv2_output, dev_y),
                   validation_data=(test_X_conv2_output, val_y))
        y_pred, is_correct, accu, f1 = bow_rf.accuracy((test_X_conv2_output, val_y), False)

        test_acc.append(val_accuracy)
        train_acc.append(dev_accuracy)
        counter += 1


    print('k折验证结果：%s' % test_acc)
    print('验证中平均准确率：%f' % np.average(test_acc[1:]))
    print('结果汇总：%s'%(test_acc+[np.average(test_acc[1:])]))
    print('-' * 80)

    fout.write('k折验证训练结果：%s\n' % train_acc)
    fout.write('k折验证测试结果：%s\n' % test_acc)
    fout.write('验证中平均准确率：%f\n' % np.average(test_acc[1:]))
    fout.write('结果汇总：%s'%(test_acc + [np.average(test_acc[1:])]))
    fout.write('-' * 80 + '\n')
    fout.flush()

    return test_acc + [np.average(test_acc[1:])],train_acc