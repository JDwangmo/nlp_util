#encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-07-14'
    Email:   '383287471@qq.com'
    Describe: 
"""
from __future__ import print_function
import numpy as np
import logging
from deep_learning.cnn.common import CnnBaseClass

class SingleChannelBowCNN(CnnBaseClass):
    """
        ## 简介：

        1. CNN(multi-channel BOW)模型,CNN-BOW(L)单通道模型，以 BOW 计数向量或 tfidf向量作为输入，以CNN为分类模型。

        2. BOW的切分粒度(feature type)有三种选择：
            - 字(word),
            - 词(seg),
            - 字词组合(word_seg)

        3. 模型架构为：
            1. 输入层： shape 为： (1, vocabulary_size ,1)
            2. reshape层：
            3. 第一层卷积层：多核卷积层:
            4. 第二层卷积层：单核卷积层
            5. flatten层
            6. 全连接层
            7. 输出Dropout层
            8. softmax分类层

    """

    def __init__(self,
                 rand_seed=1337,
                 verbose=0,
                 feature_encoder=None,
                 optimizers='sgd',
                 input_length = None,
                 num_labels=None,
                 nb_epoch=100,
                 earlyStoping_patience=50,
                 l2_conv_filter_type =None,
                 l1_conv_filter_type = None,
                 full_connected_layer_units = None,
                 output_dropout_rate = 0.,
                 **kwargs
                 ):


        super(SingleChannelBowCNN, self).__init__(
            rand_seed=rand_seed,
            verbose=verbose,
            feature_encoder=feature_encoder,
            optimizers=optimizers,
            input_length=input_length,
            num_labels=num_labels,
            nb_epoch=nb_epoch,
            earlyStoping_patience=earlyStoping_patience,
            **kwargs
        )

        self.l1_conv_filter_type = l1_conv_filter_type
        self.l2_conv_filter_type = l2_conv_filter_type
        self.full_connected_layer_units = full_connected_layer_units
        self.output_dropout_rate = output_dropout_rate

        self.build_model()

    def create_network(self):
        '''
            创建单通道的 BOW-CNN（L）神经网络
                1. 输入层：( self.input_length, )
                2. reshape层：
                3. 第一层卷积层：多核卷积层:
                4. 第二层卷积层：单核卷积层
                5. flatten层
                6. 全连接层
                7. 输出Dropout层
                8. softmax分类层

        :return:
        '''


        from keras.layers import Input, Activation, merge, Flatten, Dropout,Reshape
        from keras.models import Model
        # from keras import backend as K

        # 1. 输入层：(1,self.input_length,1)
        l1_input_shape = ( self.input_length, )

        l1_input = Input(shape=l1_input_shape)

        # 2. reshape层
        l2_reshape = Reshape((1,l1_input_shape[0],1))(l1_input)
        # 3. 多核卷积层
        l3_conv = self.create_convolution_layer(
            input_layer=l2_reshape,
            convolution_filter_type=self.l1_conv_filter_type,
            )

        # model = Model(input=l1_input, output=[l3_conv])
        # model.summary()
        # quit()
        # 4. 单核卷积层
        l4_conv = self.create_convolution_layer(
            input_layer=l3_conv,
            convolution_filter_type=self.l2_conv_filter_type,
        )
        # 5. flatten层
        l5_flatten = Flatten()(l4_conv)
        # 6. 全连接层
        l6_full_connected_layer = self.create_full_connected_layer(
            input_layer=l5_flatten,
            units=self.full_connected_layer_units + [self.num_labels]
        )
        # 7. 输出Dropout层
        l6_dropout = Dropout(p=self.output_dropout_rate)(l6_full_connected_layer)

        # 8. softmax分类层
        l8_softmax_output = Activation("softmax")(l6_dropout)

        model = Model(input=l1_input, output=[l8_softmax_output])

        if self.verbose>0:
            model.summary()

        return model

    @staticmethod
    def cross_validation(cv_data,test_data,result_file_path,**kwargs):
        '''
            进行参数的交叉验证

        :param cv_data: k份训练数据
        :type cv_data: array-like
        :param test_data: 测试数据
        :type test_data: array-like
        :return:
        '''

        nb_epoch = kwargs['nb_epoch']
        verbose = kwargs['verbose']
        num_labels = 24
        feature_type = 'word_seg'
        rand_seed = kwargs['rand_seed']
        l1_conv_filter_type = kwargs['l1_conv_filter_type']
        l2_conv_filter_type = kwargs['l2_conv_filter_type']
        k = kwargs['k']

        # 详细结果保存到...
        detail_result_file_path = result_file_path % feature_type
        fout = open(detail_result_file_path, 'w')

        print('=' * 150)
        print('feature_type:%s\nnb_epoch:%d\nrand_seed:%d' % (feature_type, nb_epoch, rand_seed))
        print('l1_conv_filter_type:%s' % l1_conv_filter_type)
        print('l2_conv_filter_type:%s' % l2_conv_filter_type)
        print('k:%s' % k)
        print('=' * 150)

        fout.write('=' * 150 + '\n')
        fout.write('single单通道CNN-bow cv结果:\n')
        fout.write('feature_type:%s\nnb_epoch:%d\nrand_seed:%d\n' % (feature_type, nb_epoch, rand_seed))
        fout.write('l1_conv_filter_type:%s\n' % l1_conv_filter_type)
        fout.write('l2_conv_filter_type:%s\n' % l2_conv_filter_type)
        fout.write('k:%s\n' % k)
        fout.write('=' * 150 + '\n')

        from data_processing_util.feature_encoder.bow_feature_encoder import FeatureEncoder
        from data_processing_util.cross_validation_util import transform_cv_data
        feature_encoder = FeatureEncoder(
            verbose=0,
            need_segmented=True,
            full_mode=True,
            remove_stopword=False,
            replace_number=True,
            lowercase=True,
            zhs2zht=True,
            remove_url=True,
            feature_method='bow',
            feature_type=feature_type,
            max_features=2000,
        )

        all_cv_data = transform_cv_data(feature_encoder,cv_data,test_data)

        for layer1 in kwargs['layer1']:
            for layer2 in kwargs['layer2']:
                for hidden1 in kwargs['hidden1']:
                    for hidden2 in kwargs['hidden2']:

                        print('layer1:%d,layer2:%d,hidden1:%d,hidden2:%d' % (layer1, layer2, hidden1, hidden2))

                        fout.write('=' * 150 + '\n')
                        fout.write('layer1:%d,layer2:%d,hidden1:%d,hidden2:%d\n' % (layer1,
                                                                                    layer2,
                                                                                    hidden1,
                                                                                    hidden2
                                                                                    ))
                        # 五折
                        print('K折交叉验证开始...')
                        counter = 0
                        ave_acc = []
                        for dev_X, dev_y, val_X, val_y in all_cv_data:
                            # print(dev_X2.shape)
                            print('-' * 80)
                            fout.write('-' * 80 + '\n')
                            if counter==0:
                                # 第一个数据是训练，之后是交叉验证
                                print('训练:' )
                                fout.write('训练\n')
                            else:
                                print('第%d个验证' % counter)
                                fout.write('第%d个验证\n' % counter)

                            bow_cnn = SingleChannelBowCNN(
                                rand_seed=1337,
                                verbose=verbose,
                                feature_encoder=None,
                                num_labels=num_labels,
                                input_length=dev_X.shape[1],
                                l1_conv_filter_type=[
                                    [layer1, l1_conv_filter_type[0], -1, 'valid', (k[0], 1), 0.5],
                                    [layer1, l1_conv_filter_type[1], -1, 'valid', (k[0], 1), 0.],
                                    [layer1, l1_conv_filter_type[2], -1, 'valid', (k[0], 1), 0.],
                                ],
                                l2_conv_filter_type=[
                                    [layer2, l2_conv_filter_type[0], -1, 'valid', (k[1], 1), 0.5]
                                ],
                                full_connected_layer_units=[hidden1, hidden2],
                                output_dropout_rate=0.2,
                                nb_epoch=nb_epoch,
                                earlyStoping_patience=50,
                                optimizers='sgd',
                                batch_size=32,
                                lr=1e-2,
                            )


                            dev_loss, dev_accuracy, \
                            val_loss, val_accuracy = bow_cnn.fit((dev_X, dev_y), (val_X, val_y))
                            print('dev:%f,%f' % (dev_loss, dev_accuracy))
                            print('val:%f,%f' % (val_loss, val_accuracy))
                            fout.write('dev:%f,%f\n' % (dev_loss, dev_accuracy))
                            fout.write('val:%f,%f\n' % (val_loss, val_accuracy))
                            ave_acc.append(val_accuracy)
                            counter += 1

                        print('k折验证结果：%s' % ave_acc)
                        print('验证中平均准确率：%f'%np.average(ave_acc[1:]))
                        print('-' * 80)

                        fout.write('k折验证结果：%s\n' % ave_acc)
                        fout.write('平均：%f\n' % np.average(ave_acc[1:]))
                        fout.write('-' * 80 + '\n')
                        fout.flush()
        fout.close()


    def print_model_descibe(self):
        import pprint
        detail = {'rand_seed': self.rand_seed,
                  'verbose': self.verbose,
                  'optimizers': self.optimizers,
                  'input_length': self.input_length,
                  'num_labels': self.num_labels,
                  'nb_epoch': self.nb_epoch,
                  'earlyStoping_patience': self.earlyStoping_patience,
                  'lr': self.lr,
                  'batch_size': self.batch_size,
                  'l2_conv_filter_type':self.l2_conv_filter_type,
                  'l1_conv_filter_type':self.l1_conv_filter_type,
                  'full_connected_layer_units':self.full_connected_layer_units,
                  'output_dropout_rate': self.output_dropout_rate,
                  }
        pprint.pprint(detail)
        logging.debug(detail)
        return detail


def test_single_bow():
    train_X = ['你好', '无聊', '测试句子', '今天天气不错', '我要买手机']
    trian_y = [1, 3, 2, 2, 3]
    test_X = ['句子,句子', '你好', '你妹']
    test_y = [2, 3, 0]
    # 生成字词组合级别的特征
    from data_processing_util.feature_encoder.bow_feature_encoder import FeatureEncoder
    feature_encoder = FeatureEncoder(
        verbose=0,
        need_segmented=True,
        full_mode=True,
        remove_stopword=True,
        replace_number=True,
        lowercase=True,
        zhs2zht=True,
        remove_url=True,
        feature_method='bow',
        feature_type='word_seg',
        max_features=2000,
    )
    train_X_feature = feature_encoder.fit_transform(train_X)
    test_X_feature = feature_encoder.transform(test_X)
    print(feature_encoder.vocabulary_size)
    print(','.join(feature_encoder.vocabulary))
    print(train_X_feature)
    print(test_X_feature)
    bow_cnn = SingleChannelBowCNN(
        rand_seed=1337,
        verbose=1,
        feature_encoder=feature_encoder,
        num_labels=4,
        input_length=feature_encoder.vocabulary_size,
        l1_conv_filter_type=[
            [5, 2, 1, 'valid', (-2, 1), 0.5],
            # [5, 4, 1, 'valid',(-2,1),0.],
            # [5, 6, 1, 'valid',(-2,1),0.],
        ],
        l2_conv_filter_type=[
            [3, 2, 1, 'valid', (-1, 1), 0.]
        ],
        full_connected_layer_units=[50, 100],
        output_dropout_rate=0.5,
        nb_epoch=30,
        earlyStoping_patience=50,
        optimizers='sgd',
        batch_size=2,
    )
    bow_cnn.print_model_descibe()
    # bow_cnn.model_from_pickle('model/AA.pkl')
    print(bow_cnn.fit(
        (train_X_feature, trian_y),
        (test_X_feature, test_y)))
    print(bow_cnn.predict('你好', transform_input=True))
    bow_cnn.accuracy((test_X_feature, test_y))
    print(bow_cnn.batch_predict(test_X, True))
    print(bow_cnn.batch_predict(test_X_feature, False))
    # bow_cnn.save_model('model/AA.pkl')

def test_cv():
    cv_x = [['你好', '无聊'],[ '测试句子', '今天天气不错'],[ '我要买手机']]
    cv_y = [[1, 3], [2, 2], [3]]
    test_x = ['你好','不错哟']
    test_y = [1,2]

    SingleChannelBowCNN.cross_validation(
        (cv_x,cv_y),
        (test_x,test_y),
        'single_%s_bow_cv_detail.txt',
        rand_seed=1337,
        nb_epoch=30,
        verbose = 0,
        feature_type = 'word_seg',
        layer1 = [3, 5],
        l1_conv_filter_type = [2,3,4],
        layer2 = [3,7],
        l2_conv_filter_type = [5],
        k = [2, 2],
        hidden1 = [50, 100],
        hidden2 = [50, 100],

    )

if __name__ == '__main__':

    # test_single_bow()
    test_cv()




