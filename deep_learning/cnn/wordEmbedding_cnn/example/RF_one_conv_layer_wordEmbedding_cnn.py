# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-08-02'
    Email:   '383287471@qq.com'
    Describe: RF(CNN(static-w2v)) ,使用 单层卷积层的 CNN（static-w2v） 提取特征，RF做分类器
        - 输入层
        -
"""

from traditional_classify.bow_rf.bow_rf_model import BowRandomForest
from deep_learning.cnn.wordEmbedding_cnn.wordEmbedding_cnn_model import WordEmbeddingCNN

# 实现混合模型 RF(CNN(w2v))
class RFAndWordEmbeddingCnnMerge(object):
    def __init__(self,
                 feature_encoder,
                 num_filter,
                 num_labels,
                 n_estimators,
                 word2vec_model_file_path,
                 **kwargs
                 ):

        self.static_w2v_cnn = WordEmbeddingCNN(
            rand_seed=1377,
            verbose=kwargs.get('verbose', 0),
            feature_encoder=feature_encoder,
            optimizers='sgd',
            word_embedding_dim=50,
            # 设置embedding使用训练好的w2v模型初始化
            embedding_init_weight=feature_encoder.to_embedding_weight(word2vec_model_file_path),
            # 设置为训练时embedding层权重不变
            embedding_weight_trainable=False,
            num_labels=num_labels,
            l1_conv_filter_type=[
                [num_filter, 3, -1, 'valid', (-1, 1), 0., 'relu', 'none'],
                [num_filter, 4, -1, 'valid', (-1, 1), 0., 'relu', 'none'],
                [num_filter, 5, -1, 'valid', (-1, 1), 0., 'relu', 'none'],
            ],
            l2_conv_filter_type=[],
            full_connected_layer_units=[],
            embedding_dropout_rate=0.,
            nb_epoch=30,
            nb_batch=32,
            earlyStoping_patience=30,
            lr=1e-2,
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

        self.static_w2v_cnn.fit(train_data,validation_data)

        train_x_features = self.static_w2v_cnn.get_layer_output(train_X,
                                                                layer='conv1',
                                                                transform_input=False
                                                                )

        validation_x_features = self.static_w2v_cnn.get_layer_output(validation_X,
                                                                layer='conv1',
                                                                transform_input=False
                                                                )

        return self.bow_randomforest.fit((train_x_features,train_y),(validation_x_features,validation_y))


class RFAndWordEmbeddingCNNWithOneConv(object):

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
            input_length =None,
            num_filter_list=None,
            n_estimators_list=None,
            verbose = 0,
            word2vec_model_file_path = None,
           ):

        from data_processing_util.cross_validation_util import transform_cv_data, get_k_fold_data, get_val_score
        # 1. 获取交叉验证的数据
        if cv_data is None:
            assert train_data is not None, 'cv_data和train_data必须至少提供一个！'
            cv_data = get_k_fold_data(
                k=3,
                train_data=train_data,
                test_data=test_data,
                include_train_data=True,
            )

        # 2. 将数据进行特征编码转换
        feature_encoder = WordEmbeddingCNN.get_feature_encoder(
            input_length=input_length,
            verbose=0,
            feature_type='word',
        )
        cv_data = transform_cv_data(feature_encoder, cv_data,verbose=0)
        # 交叉验证
        for num_filter in num_filter_list:
            for n_estimators in n_estimators_list:
                print('=' * 40)
                print('num_filter and n_estimators is %d,%d.'%(num_filter,n_estimators))
                get_val_score(RFAndWordEmbeddingCNNWithOneConv,
                              cv_data=cv_data,
                              verbose=verbose,
                              num_filter=num_filter,
                              n_estimators = n_estimators,
                              num_labels=24,
                              word2vec_model_file_path = word2vec_model_file_path,
                              )


if __name__ == '__main__':
    train_x = ['你好', '测试句子', '我要买手机', '今天天气不错', '无聊']
    train_y = [1, 2, 3, 2, 3]
    test_x = ['你好', '不错哟']
    test_y = [1, 2]
    cv_x = [['你好', '无聊'], ['测试句子', '今天天气不错'], ['我要买手机']]
    cv_y = [[1, 3], [2, 2], [3]]

    RFAndWordEmbeddingCNNWithOneConv.cross_validation(
        train_data = (train_x,train_y),
        test_data=(test_x,test_y),
        input_length=8,
        num_filter_list=[5,50],
        n_estimators_list = [10],
        verbose=1,
        word2vec_model_file_path = '/home/jdwang/PycharmProjects/corprocessor/word2vec/vector/50dim/vector1000000_50dim.gem'

    )