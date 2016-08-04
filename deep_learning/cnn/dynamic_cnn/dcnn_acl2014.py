# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-08-03'
    Email:   '383287471@qq.com'
    Describe: ACL 2014 版本 的 DCNN，具体参考：
"""

from dcnn_model import DCNN

class DcnnAcl(object):
    @staticmethod
    def get_model(
            feature_encoder,
            conv1_num_filter,
            conv2_num_filter,
            num_labels,
            **kwargs
    ):
        dcnn = DCNN(
            rand_seed=1377,
            verbose=1,
            feature_encoder=feature_encoder,
            # optimizers='adadelta',
            optimizers='sgd',
            input_dim=feature_encoder.vocabulary_size,
            word_embedding_dim=50,
            # 设置embedding使用训练好的w2v模型初始化
            embedding_init_weight=None,
            # 设置为训练时embedding层权重可变
            embedding_weight_trainable=True,
            num_labels=num_labels,
            l1_conv_filter_type=[
                [conv1_num_filter, 3, -1, '1D', (-6, 1), 0.,'relu','none'],
                # [4, 3, -1, '1D', (-2, 1), 0., 'none', 'none'],
            ],
            l2_conv_filter_type=[
                [conv2_num_filter, 2, -1, '1D', (-3, 1), 0.1,'relu','batch_normalization'],
            ],
            full_connected_layer_units=[
                # [50]
            ],
            embedding_dropout_rate=0.,
            nb_epoch=30,
            nb_batch=32,
            earlyStoping_patience=30,
            lr=1e-2,
        )

        return dcnn

    @staticmethod
    def cross_validation(
            train_data=None,
            test_data=None,
            cv_data=None,
            input_length=None,
            conv1_num_filter_list=None,
            conv2_num_filter_list=None,
            verbose=0,
            word2vec_model_file_path=None,
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
        feature_encoder = DCNN.get_feature_encoder(
            input_length=input_length,
            verbose=0,
            full_mode=False,
            feature_type='word',
        )
        cv_data = transform_cv_data(feature_encoder, cv_data, verbose=0)
        # 交叉验证
        for conv1_num_filter in conv1_num_filter_list:
            for conv2_num_filter in conv2_num_filter_list:
                print('=' * 40)
                print('num_filter of conv1 and conv2 is %d,%d .' % (conv1_num_filter,conv2_num_filter))
                get_val_score(DcnnAcl,
                              cv_data=cv_data,
                              verbose=verbose,
                              conv1_num_filter=conv1_num_filter,
                              conv2_num_filter=conv2_num_filter,
                              num_labels=24,
                              word2vec_model_file_path=word2vec_model_file_path,
                              )


if __name__ == '__main__':
    train_x = ['你好', '测试句子', '我要买手机', '今天天气不错', '无聊']
    train_y = [1, 2, 3, 2, 3]
    test_x = ['你好', '不错哟']
    test_y = [1, 2]
    cv_x = [['你好', '无聊'], ['测试句子', '今天天气不错'], ['我要买手机']]
    cv_y = [[1, 3], [2, 2], [3]]

    DcnnAcl.cross_validation(
        train_data=(train_x, train_y),
        test_data=(test_x, test_y),
        input_length=8,
        conv1_num_filter_list=[5],
        conv2_num_filter_list=[3],
        verbose=1,
    )