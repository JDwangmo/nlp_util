# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-08-02'
    Email:   '383287471@qq.com'
    Describe: 单层卷积层的 CNN（static-w2v） or CNN(non-static-w2v) 模型
        - 输入层
        - valid convolution layer： 多size
        - 1-max pooling： 句子方向
        - softmax output layer

    Notes


    更多可参考： https://github.com/JDwangmo/coprocessor/tree/master/reference#2convolutional-neural-networks-for-sentence-classification

"""

from deep_learning.cnn.wordEmbedding_cnn.wordEmbedding_cnn_model import WordEmbeddingCNN


class WordEmbeddingCNNWithOneConv(object):
    # 如果使用全体数据作为字典，则使用这个变量来存放权重，避免重复加载权重，因为每次加载的权重都是一样的。
    train_data_weight = None
    # 验证数据是一份权重，不包含测试集了
    val_data_weight = None

    @staticmethod
    def get_model(
            feature_encoder,
            num_filter,
            num_labels,
            word2vec_model_file_path,
            **kwargs
    ):
        # print(WordEmbeddingCNNWithOneConv.weight)
        if kwargs['dataset_flag']==0:
            if WordEmbeddingCNNWithOneConv.train_data_weight is None:
                # 训练集
                WordEmbeddingCNNWithOneConv.train_data_weight = feature_encoder.to_embedding_weight(word2vec_model_file_path)

            weight = WordEmbeddingCNNWithOneConv.train_data_weight

        else:
            # kwargs['dataset_flag']>0
            if WordEmbeddingCNNWithOneConv.val_data_weight is None:
                WordEmbeddingCNNWithOneConv.val_data_weight = feature_encoder.to_embedding_weight(word2vec_model_file_path)
            weight = WordEmbeddingCNNWithOneConv.val_data_weight
        print(weight)
        static_w2v_cnn = WordEmbeddingCNN(
            rand_seed=1377,
            verbose=kwargs.get('verbose', 0),
            feature_encoder=feature_encoder,
            # optimizers='adadelta',
            optimizers='sgd',
            # word_embedding_dim=300,
            # 设置embedding使用训练好的w2v模型初始化
            embedding_init_weight=weight,
            # 设置为训练时embedding层权重不变
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
            nb_epoch=30,
            nb_batch=32,
            earlyStoping_patience=30,
            lr=1e-2,
            show_validate_accuracy=True if kwargs.get('verbose', 0) > 0 else False,
            # output_regularizer=('l2', 0.5),
            output_constraints=('maxnorm', 3),
        )
        # static_w2v_cnn.print_model_descibe()
        # quit()
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
            need_segmented=True,
            word2vec_model_file_path=None,
            num_labels=24,
            embedding_weight_trainable=False,
            need_validation=True,
            include_train_data=True,
            vocabulary_including_test_set=True,

    ):
        """

        Parameters
        ----------
        num_labels: int
            标签
        vocabulary_including_test_set: bool,default,True
            字典是否包括测试集
        include_train_data: 是否包含训练数据一样验证
        need_validation: 是否要验证
        embedding_weight_trainable: 切换 CNN(static-w2v) 和 CNN(non-static-w2v)
        train_data:
        test_data
        cv_data
        feature_type
        input_length
        num_filter_list
        verbose
        cv:int
            进行 cv 折验证
        need_segmented:bool
            是否需要分词
        word2vec_model_file_path

        Notes
        ----------
        - 为了提高效率，默认设置 update_dictionary = True ,以保证feature encoder的字典一致，避免重复构造字典
        - 同时设置 diff_train_val_feature_encoder=True来保证训练集上和验证集上的feature encoder 不同，因为字典大小不同

        Examples
        ----------
        >>> train_x = ['你好', '测试句子', '我要买手机', '今天天气不错', '无聊']
        >>> train_y = [1, 2, 3, 2, 3]
        >>> test_x = ['你好', '不错哟']
        >>> test_y = [1, 2]
        >>> cv_x = [['你好', '无聊'], ['测试句子', '今天天气不错'], ['我要买手机']]
        >>> cv_y = [[1, 3], [2, 2], [3]]
        >>> WordEmbeddingCNNWithOneConv.cross_validation(
        >>>         train_data = (train_x,train_y),
        >>>         test_data=(test_x,test_y),
        >>>         input_length=8,
        >>>         num_filter_list=[5,50],
        >>>         verbose=1,
        >>>         word2vec_model_file_path = '/home/jdwang/PycharmProjects/nlp_util/data_processing_util/word2vec_util/vector/50dim/vector1000000_50dim.gem',
        >>>     )

        """
        print('=' * 80)
        print('feature_type:%s,need_segmented:%s,vocabulary_including_test_set:%s' % (feature_type,
                                                                                      need_segmented,
                                                                                      vocabulary_including_test_set))
        print('embedding_weight_trainable:%s' % embedding_weight_trainable)
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

        cv_data = transform_cv_data(feature_encoder, cv_data, verbose=verbose, diff_train_val_feature_encoder=True)
        # 交叉验证
        for num_filter in num_filter_list:
            print('=' * 40)
            print('num_filter is %d.' % num_filter)
            get_val_score(WordEmbeddingCNNWithOneConv,
                          cv_data=cv_data[:],
                          verbose=verbose,
                          num_filter=num_filter,
                          num_labels=num_labels,
                          word2vec_model_file_path=word2vec_model_file_path,
                          embedding_weight_trainable=embedding_weight_trainable,
                          need_validation=need_validation,
                          )


if __name__ == '__main__':
    train_x = ['你好', '测试句子', '我要买手机', '今天天气不错', '无聊']
    train_y = [1, 2, 3, 2, 3]
    test_x = ['你好', '不错哟']
    test_y = [1, 2]
    cv_x = [['你好', '无聊'], ['测试句子', '今天天气不错'], ['我要买手机']]
    cv_y = [[1, 3], [2, 2], [3]]

    WordEmbeddingCNNWithOneConv.cross_validation(
        train_data=(train_x, train_y),
        test_data=(test_x, test_y),
        input_length=8,
        num_filter_list=[5, 50],
        verbose=1,
        word2vec_model_file_path='/home/jdwang/PycharmProjects/corprocessor/word2vec/vector/50dim/vector1000000_50dim.gem',
    )
