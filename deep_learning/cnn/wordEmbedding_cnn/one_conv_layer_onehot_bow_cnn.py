# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-08-01'
    Email:   '383287471@qq.com'
    Describe: 构架更特定的 CNN-bow模型，本脚本构建一个 只有一层卷积层的 CNN-bow模型
        - 输入层
        - bow convolution ： 类似于一个 嵌入层，不同之处，在于将 region of sentence（onehot表示） 转为一个 低维 real-value 向量。
        - max pooling layer
        - softmax output layer

    更多可参考： https://github.com/JDwangmo/coprocessor/tree/master/reference#3effective-use-of-word-order-for-text-categorization-with-convolutional-neural-networks

"""

from onehot_cnn_model import OnehotBowCNN

class OnehotBowCNNWithOneConv(object):

    @staticmethod
    def get_model(
            feature_encoder,
            num_filter,
            region_size,
            num_labels,
            **kwargs
    ):
        onehot_cnn = OnehotBowCNN(
            rand_seed=1377,
            verbose=kwargs.get('verbose',0),
            feature_encoder=feature_encoder,
            # optimizers='adadelta',
            optimizers='sgd',
            num_labels=num_labels,
            l1_conv_filter_type=[
                [num_filter, region_size, -1, 'bow', (-1, 1), 0., 'relu', 'batch_normalization'],
            ],
            l2_conv_filter_type=[
            ],
            full_connected_layer_units=[
            ],
            embedding_dropout_rate=0.,
            nb_epoch=30,
            nb_batch=32,
            earlyStoping_patience=30,
            lr=1e-2,
        )

        return onehot_cnn

    @staticmethod
    def cross_validation(
            train_data=None,
            test_data=None,
            cv_data=None,
            input_length =None,
            num_filter_list=None,
            region_size_list=None,
            word2vec_to_solve_oov = False,
            word2vec_model_file_path = None,
            verbose = 0,
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
        feature_encoder = OnehotBowCNN.get_feature_encoder(
            input_length=input_length,
            verbose=verbose,
            feature_type='word',
            word2vec_to_solve_oov = word2vec_to_solve_oov,
            word2vec_model_file_path=word2vec_model_file_path,
        )
        cv_data = transform_cv_data(feature_encoder, cv_data,verbose=0)
        # 交叉验证
        for num_filter in num_filter_list:
            for region_size in region_size_list:
                print('=' * 40)
                print('num_filter and region_size is %d,%d.'%(num_filter,region_size))
                get_val_score(OnehotBowCNNWithOneConv,
                              cv_data=cv_data,
                              verbose=verbose,
                              region_size = region_size,
                              num_filter=num_filter,
                              num_labels=24
                              )


if __name__ == '__main__':
    train_x = ['你好', '测试句子', '我要买手机', '今天天气不错', '无聊']
    train_y = [1, 2, 3, 2, 3]
    test_x = ['你好', '不错哟']
    test_y = [1, 2]
    cv_x = [['你好', '无聊'], ['测试句子', '今天天气不错'], ['我要买手机']]
    cv_y = [[1, 3], [2, 2], [3]]

    OnehotBowCNNWithOneConv.cross_validation(
        train_data = (train_x,train_y),
        test_data=(test_x,test_y),
        input_length=8,
        num_filter_list=[5,50],
        region_size_list=range(1,9),
        verbose=0,
        word2vec_to_solve_oov=True,
        word2vec_model_file_path = '/home/jdwang/PycharmProjects/corprocessor/word2vec/vector/50dim/vector1000000_50dim.gem'

    )