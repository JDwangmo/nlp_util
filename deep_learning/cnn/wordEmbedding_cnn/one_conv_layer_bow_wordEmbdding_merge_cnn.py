# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-08-02'
    Email:   '383287471@qq.com'
    Describe: 单层卷积层 CNN（w2v+boc）
"""



from onehot_wordEmbedding_merge_cnn_model import BowWordEmbeddingMergeCNN



class BowWordEmbeddingMergeCNNWithOneConv(object):

    @staticmethod
    def get_model(
            feature_encoder,
            w2v_num_filter,
            bow_num_filter,
            bow_region_size,
            num_labels,
            word2vec_model_file_path,
            **kwargs
    ):

        merge_cnn = BowWordEmbeddingMergeCNN(
            rand_seed=1377,
            verbose=kwargs.get('verbose',0),
            feature_encoder=feature_encoder,
            # optimizers='adadelta',
            optimizers='sgd',
            word_embedding_dim=50,
            # 设置embedding使用训练好的w2v模型初始化
            init_embedding_weight=True,
            embedding_weight_path=word2vec_model_file_path,
            # 设置为训练时embedding层权重不变
            embedding_weight_trainable=False,
            num_labels=num_labels,
            l1_w2v_conv_filter_type=[
                [w2v_num_filter, 3, -1, 'valid', (-1, 1), 0., 'relu', 'none'],
                [w2v_num_filter, 4, -1, 'valid', (-1, 1), 0., 'relu', 'none'],
                [w2v_num_filter, 5, -1, 'valid', (-1, 1), 0., 'relu', 'none'],
            ],
            l1_bow_conv_filter_type=[
                [bow_num_filter, bow_region_size, -1, 'bow', (-1, 1), 0., 'relu', 'none'],
            ],
            l2_conv_filter_type=[],
            full_connected_layer_units=[],
            embedding_dropout_rate=0.,
            nb_epoch=30,
            nb_batch=32,
            earlyStoping_patience=30,
            lr=1e-2,
        )

        return merge_cnn

    @staticmethod
    def cross_validation(
            train_data=None,
            test_data=None,
            cv_data=None,
            input_length =None,
            bow_num_filter_list=None,
            w2v_num_filter_list=None,
            bow_region_size_list = None,
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
        feature_encoder = BowWordEmbeddingMergeCNN.get_feature_encoder(
            input_length=input_length,
            verbose=0,
            feature_type='word',
        )


        cv_data = transform_cv_data(feature_encoder, cv_data,verbose=0)
        # 交叉验证
        for bow_num_filter in bow_num_filter_list:
            for bow_region_size in bow_region_size_list:
                for w2v_num_filter in w2v_num_filter_list:

                    print('=' * 40)
                    print('bow_num_filter,bow_region_size and w2v_num_filter is %d,%d,%d.'%(bow_num_filter,bow_region_size,w2v_num_filter))
                    get_val_score(BowWordEmbeddingMergeCNNWithOneConv,
                                  cv_data=cv_data,
                                  verbose=verbose,
                                  bow_num_filter = bow_num_filter,
                                  bow_region_size = bow_region_size,
                                  w2v_num_filter = w2v_num_filter,
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

    BowWordEmbeddingMergeCNNWithOneConv.cross_validation(
        train_data = (train_x,train_y),
        test_data=(test_x,test_y),
        input_length=8,
        bow_num_filter_list=[5,50],
        bow_region_size_list = [3,14],
        w2v_num_filter_list=[5,50],
        verbose=0,
        word2vec_model_file_path = '/home/jdwang/PycharmProjects/corprocessor/word2vec/vector/50dim/vector1000000_50dim.gem'

    )