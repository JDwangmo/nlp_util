# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-08-02'; 'last updated date: 2016-09-28'
    Email:   '383287471@qq.com'
    Describe: RF(CNN(static-w2v)) ,使用 单层卷积层的 CNN（static-w2v） 提取特征，RF做分类器
        - 输入层
        -
"""

from traditional_classify.bow_rf.bow_rf_model import BowRandomForest
from deep_learning.cnn.wordEmbedding_cnn.wordEmbedding_cnn_model import WordEmbeddingCNN


# 实现混合模型 RF(CNN(w2v))
class RFAndWordEmbeddingCnnMerge(object):
    # 如果使用全体数据作为字典，则使用这个变量来存放权重，避免重复加载权重，因为每次加载的权重都是一样的。
    train_data_weight = None
    # 验证数据是一份权重，不包含测试集了
    val_data_weight = None

    def __init__(self,
                 feature_encoder,
                 num_filter,
                 num_labels,
                 n_estimators,
                 word2vec_model_file_path,
                 **kwargs
                 ):

        if kwargs.get('rand_weight', False):
            # CNN(rand)模式
            weight = None
        elif kwargs['dataset_flag'] == 0:
            if RFAndWordEmbeddingCnnMerge.train_data_weight is None:
                # 训练集
                RFAndWordEmbeddingCnnMerge.train_data_weight = feature_encoder.to_embedding_weight(
                    word2vec_model_file_path)
            weight = RFAndWordEmbeddingCnnMerge.train_data_weight
        else:
            # kwargs['dataset_flag']>0
            if RFAndWordEmbeddingCnnMerge.val_data_weight is None:
                RFAndWordEmbeddingCnnMerge.val_data_weight = feature_encoder.to_embedding_weight(
                    word2vec_model_file_path)
            weight = RFAndWordEmbeddingCnnMerge.val_data_weight
        # print(weight)
        self.static_w2v_cnn = WordEmbeddingCNN(
            rand_seed=1377,
            verbose=kwargs.get('verbose', 0),
            feature_encoder=feature_encoder,
            # optimizers='adadelta',
            optimizers='sgd',
            # 当使用CNN (rand) 模式的时候使用到了
            word_embedding_dim=50,
            # 设置embedding使用训练好的w2v模型初始化
            embedding_init_weight=weight,
            # 默认设置为训练时embedding层权重不变
            embedding_weight_trainable=kwargs.get('embedding_weight_trainable', False),
            num_labels=num_labels,
            l1_conv_filter_type=[
                [num_filter, 3, -1, 'valid', (-1, 1), 0.5, 'relu', 'none'],
                [num_filter, 4, -1, 'valid', (-1, 1), 0., 'relu', 'none'],
                [num_filter, 5, -1, 'valid', (-1, 1), 0., 'relu', 'none'],
            ],
            l2_conv_filter_type=[],
            full_connected_layer_units=[],
            embedding_dropout_rate=0.,
            nb_epoch=kwargs.get('nb_epoch', 25),
            batch_size=kwargs.get('batch_size', 32),
            earlyStoping_patience=30,
            lr=kwargs.get('lr', 1e-2),
            show_validate_accuracy=True if kwargs.get('verbose', 0) > 0 else False,
            # output_regularizer=('l2', 0.5),
            output_constraints=('maxnorm', 3),
            # 必须设为True，才能取中间结果做特征
            save_middle_output=True,

        )

        self.bow_randomforest = BowRandomForest(
            rand_seed=1377,
            verbose=kwargs.get('verbose', 0),
            feature_encoder=feature_encoder,
            # optimizers='adadelta',
            n_estimators=n_estimators,
            min_samples_leaf=1,
        )

    def fit(self, train_data=None, validation_data=None):
        train_X, train_y = train_data
        validation_X, validation_y = validation_data

        self.static_w2v_cnn.fit(train_data, validation_data)

        train_x_features = self.static_w2v_cnn.get_layer_output(train_X)[4]

        validation_x_features = self.static_w2v_cnn.get_layer_output(validation_X)[4]

        return self.bow_randomforest.fit((train_x_features, train_y), (validation_x_features, validation_y))


class RFAndRFAndWordEmbeddingCnnMerge(object):
    @staticmethod
    def get_model(
            feature_encoder,
            num_filter,
            num_labels,
            n_estimators,
            word2vec_model_file_path,
            **kwargs
    ):

        static_w2v_cnn = RFAndWordEmbeddingCnnMerge(
            feature_encoder,
            num_filter,
            num_labels,
            n_estimators,
            word2vec_model_file_path,
            **kwargs
        )

        return static_w2v_cnn

    @staticmethod
    def cross_validation(
            train_data=None,
            test_data=None,
            cv_data=None,
            feature_type='word',
            input_length=None,
            num_filter_list=None,
            verbose=0,
            cv=3,
            batch_size=32,
            lr=1e-2,
            need_segmented=True,
            word2vec_model_file_path=None,
            num_labels=24,
            embedding_weight_trainable=False,
            shuffle_data=True,
            rand_weight=False,
            need_validation=True,
            include_train_data=True,
            vocabulary_including_test_set=True,
            n_estimators_list=None,
    ):

        print('=' * 80)
        print('feature_type:%s,need_segmented:%s,vocabulary_including_test_set:%s' % (feature_type,
                                                                                      need_segmented,
                                                                                      vocabulary_including_test_set))
        print('rand_weight:%s,embedding_weight_trainable:%s' % (rand_weight, embedding_weight_trainable))
        print('=' * 80)

        from data_processing_util.cross_validation_util import transform_cv_data, get_k_fold_data, get_val_score

        # 1. 获取交叉验证的数据
        if cv_data is None:
            assert train_data is not None, 'cv_data和train_data必须至少提供一个！'
            cv_data = get_k_fold_data(
                k=cv,
                train_data=train_data,
                test_data=test_data,
                include_train_data=include_train_data,
            )

        # 2. 将数据进行特征编码转换
        feature_encoder = WordEmbeddingCNN.get_feature_encoder(
            need_segmented=need_segmented,
            input_length=input_length,
            verbose=1,
            feature_type=feature_type,
            # 设置字典保持一致
            update_dictionary=False,
            vocabulary_including_test_set=vocabulary_including_test_set,
        )

        cv_data = transform_cv_data(feature_encoder, cv_data, verbose=verbose, diff_train_val_feature_encoder=1)

        # 交叉验证
        for num_filter in num_filter_list:
            for n_estimators in n_estimators_list:
                print('=' * 40)
                print('num_filter and n_estimators is %d,%d.' % (num_filter, n_estimators))
                get_val_score(RFAndRFAndWordEmbeddingCnnMerge,
                              num_filter=num_filter,
                              n_estimators=n_estimators,
                              cv_data=cv_data[:],
                              verbose=verbose,
                              num_labels=num_labels,
                              word2vec_model_file_path=word2vec_model_file_path,
                              embedding_weight_trainable=embedding_weight_trainable,
                              need_validation=need_validation,
                              rand_weight=rand_weight,
                              batch_size=batch_size,
                              lr=lr,
                              shuffle_data=shuffle_data,
                              )


if __name__ == '__main__':
    train_x = ['你好', '测试句子', '我要买手机', '今天天气不错', '无聊']
    train_y = [1, 2, 3, 2, 3]
    test_x = ['你好', '不错哟']
    test_y = [1, 2]
    cv_x = [['你好', '无聊'], ['测试句子', '今天天气不错'], ['我要买手机']]
    cv_y = [[1, 3], [2, 2], [3]]

    RFAndRFAndWordEmbeddingCnnMerge.cross_validation(
        train_data=(train_x, train_y),
        test_data=(test_x, test_y),
        input_length=8,
        num_filter_list=[5, 50],
        n_estimators_list=[10],
        verbose=1,
        word2vec_model_file_path='/home/jdwang/PycharmProjects/corprocessor/word2vec/vector/50dim/vector1000000_50dim.gem'

    )
