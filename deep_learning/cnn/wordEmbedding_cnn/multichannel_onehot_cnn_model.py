# encoding=utf8

__author__ = 'jdwang'
__date__ = 'create date: 2016-06-23'
__email__ = '383287471@qq.com'

import numpy as np
from deep_learning.cnn.common import CnnBaseClass
from itertools import product
import logging


class MultiChannelOnehotBowCNN(CnnBaseClass):
    '''
        多输入（通道）的CNN onehot模型
        架构各个层次分别为: 输入层,卷积层,1-max pooling层,全连接层,dropout层,softmax层
        具体见:
            https://github.com/JDwangmo/coprocessor#2convolutional-neural-networks-for-sentence-classification
        包含以下主要函数：
            1. build_model：构建模型
            2. fit：拟合和训练模型
            3. transform：对输入进行转换
            4. predict： 单句预测
            5. batch_predict： 批量预测
            6. save_model：保存模型
            7. model_from_pickle：恢复模型
            8. accuracy：模型验证
            9. print_model_descibe：打印模型详情
    '''

    def __init__(self,
                 rand_seed=1337,
                 verbose=0,
                 feature_encoder=None,
                 full_connected_layer_units=None,
                 optimizers='sgd',
                 word_input_length=None,
                 seg_input_length=None,
                 word_input_dim = None,
                 seg_input_dim = None,
                 num_labels=None,
                 l1_conv_filter_type=None,
                 l2_conv_filter_type=None,
                 nb_epoch=100,
                 earlyStoping_patience=50,
                 **kwargs
                 ):
        '''
            1. 初始化参数，并检验参数合法性。
            2. 设置随机种子，构建模型

        :param rand_seed: 随机种子,假如设置为为None时,则随机取随机种子
        :type rand_seed: int
        :param verbose: 数值越大,输出更详细的信息
        :type verbose: int
        :param feature_encoder: 输入数据的设置选项，设置输入编码器,(word_feature_encoder,seg_feature_encoder)
        :type feature_encoder: onehot_feature_encoder.FeatureEncoder
        :param optimizers: 求解的优化算法，目前支持: ['sgd','adadelta']
        :type optimizers: str
        :param input_length: cnn设置选项,输入句子(序列)的长度.
        :type input_length: int
        :param input_dim: cnn设置选项,onehot array dim .
        :type input_dim: int
        :param num_labels: cnn设置选项,最后输出层的大小,即分类类别的个数.
        :type num_labels: int
        :param l1_conv_filter_type: cnn设置选项,卷积层的类型.

            for example:每个列表代表一种类型(size)的卷积核,
                l1_conv_filter_type = [[100,2,word_embedding_dim,'valid',(1,1)],
                                    [100,4,word_embedding_dim,'valid',(1,1)],
                                    [100,6,word_embedding_dim,'valid',(1,1)],
                                   ]

        :type l1_conv_filter_type: array([])
        :param nb_epoch: cnn设置选项,cnn迭代的次数.
        :type nb_epoch: int
        :param earlyStoping_patience: cnn设置选项,earlyStoping的设置,如果迭代次数超过这个耐心值,依旧不下降,则stop.
        :type earlyStoping_patience: int
        '''
        CnnBaseClass.__init__(
            self,
            rand_seed=rand_seed,
            verbose=verbose,
            feature_encoder=feature_encoder,
            optimizers=optimizers,
            input_length=None,
            num_labels=num_labels,
            nb_epoch=nb_epoch,
            earlyStoping_patience=earlyStoping_patience,
        )
        if full_connected_layer_units is None:
            full_connected_layer_units = [(50, 0.5, 'relu', 'none')]

        assert word_input_length is not None or feature_encoder is not None,'word_input_length和feature_encoder至少要有一个不为None'
        assert seg_input_length is not None or feature_encoder is not None,'seg_input_length和feature_encoder至少要有一个不为None'
        assert word_input_dim is not None or feature_encoder is not None,'word_input_dim和feature_encoder至少要有一个不为None'
        assert seg_input_dim is not None or feature_encoder is not None,'seg_input_dim和feature_encoder至少要有一个不为None'
        self.feature_encoder = feature_encoder

        if feature_encoder !=None:
            # 如果feature encoder 不为空，直接用 feature_encoder获取 长度和维度
            assert len(feature_encoder)==2,'feature_encoder的输入应该是(word_feature_encoder,seg_feature_encoder)'

            self.word_input_length = feature_encoder[0].sentence_padding_length
            self.word_input_dim = feature_encoder[0].vocabulary_size

            self.seg_input_length = feature_encoder[1].sentence_padding_length
            self.seg_input_dim = feature_encoder[1].vocabulary_size

        else:

            self.word_input_length = word_input_length
            self.word_input_dim = word_input_dim
            self.seg_input_length = seg_input_length
            self.seg_input_dim = seg_input_dim

        self.l1_conv_filter_type = l1_conv_filter_type
        self.l2_conv_filter_type = l2_conv_filter_type
        self.full_connected_layer_units = full_connected_layer_units
        self.kwargs = kwargs

        # 构建模型
        self.build_model()

    def create_network(self):
        '''
            1. 创建 CNN 网络

                1. 输入层，2D，（n_batch,input_length,input_dim）
                2. Reshape层： 将embedding转换4-dim的shape，4D
                3. 第一层多size卷积层（含1-max pooling），使用三种size.
                4. Flatten层： 卷积的结果进行拼接,变成一列隐含层
                5. output hidden层
                6. output Dropout层
                7. softmax 分类层
            2. compile模型

        :return: cnn model network
        '''

        from keras.layers import Input, Activation, Reshape, merge, Flatten
        from keras.models import Model
        from deep_learning.cnn.custom_layers import MaxPooling2DWrapper
        # from keras import backend as K

        # 1.1 字输入层
        l1_word_input_shape = ( self.word_input_length,self.word_input_dim)
        l1_word_input = Input(shape=l1_word_input_shape,name='l1_word_input')

        # 2.1 字Reshape层： 将embedding转换4-dim的shape
        l2_word_reshape_output_shape = (1, l1_word_input_shape[0], l1_word_input_shape[1])
        l2_word_reshape= Reshape(l2_word_reshape_output_shape,name='l2_word_reshape')(l1_word_input)

        # 1.2 词输入层
        l1_seg_input_shape = (self.seg_input_length, self.seg_input_dim)
        l1_seg_input = Input(shape=l1_seg_input_shape,name='l1_seg_input')

        # 2.2 词Reshape层： 将embedding转换4-dim的shape
        l2_reshape_seg_output_shape = (1, l1_seg_input_shape[0], l1_seg_input_shape[1])
        l2_seg_reshape = Reshape(l2_reshape_seg_output_shape,name='l2_seg_reshape')(l1_seg_input)

        # 3.1 字 卷积层：多size卷积层（含1-max pooling），使用三种size.
        l3_word_conv = self.create_convolution_layer(
            input_layer=l2_word_reshape,
            convolution_filter_type=self.l1_conv_filter_type,
        )

        # 3.2. 词 卷积层：多size卷积层（含1-max pooling），使用三种size
        l3_seg_conv = self.create_convolution_layer(
            input_layer=l2_seg_reshape,
            convolution_filter_type=self.l1_conv_filter_type,
        )

        # 4、将两个卷积的结果合并
        l4_merge = merge([l3_word_conv,l3_seg_conv],mode='concat',concat_axis=2,name='l4_merge')

        # model = Model(input=[l1_word_input,l1_seg_input], output=[l4_merge])
        # model.summary()
        # quit()
        # 5、max pooling
        l5_pooling = MaxPooling2DWrapper((2,1),name='l5_pooling')(l4_merge)
        # 6. Flatten层： 卷积的结果进行拼接,变成一列隐含层
        l6_flatten = Flatten(name='l6_flatten')(l5_pooling)
        # 7. 全连接层
        l7_full_connected_layer = self.create_full_connected_layer(
            input_layer=l6_flatten,
            units=self.full_connected_layer_units
        )

        l8_output_layer = self.create_full_connected_layer(
            input_layer=l7_full_connected_layer,
            units=[[self.num_labels, 0., 'none', 'none']]
        )

        # 8. softmax分类层
        l9_softmax_output = Activation("softmax")(l8_output_layer)

        model = Model(input=[l1_word_input,l1_seg_input], output=[l9_softmax_output])

        if self.verbose > 0:
            model.summary()

        return model

    @staticmethod
    def get_feature_encoder(**kwargs):
        '''
            获取该分类器的特征编码器

        :param kwargs:  word_input_length,seg_input_length
        :return:
        '''

        from data_processing_util.feature_encoder.onehot_feature_encoder import FeatureEncoder
        word_feature_encoder = FeatureEncoder(
            sentence_padding_length=kwargs['word_input_length'],
            verbose=0,
            need_segmented=True,
            full_mode=False,
            replace_number=True,
            remove_stopword=True,
            lowercase=True,
            padding_mode='left',
            add_unkown_word=True,
            feature_type='word',
            zhs2zht=True,
            remove_url=True,
            # 设置为True，输出 onehot array
            to_onehot_array=True,
        )

        seg_feature_encoder = FeatureEncoder(
            sentence_padding_length=kwargs['seg_input_length'],
            verbose=0,
            need_segmented=True,
            full_mode=False,
            replace_number=True,
            remove_stopword=True,
            lowercase=True,
            padding_mode='left',
            add_unkown_word=True,
            feature_type='seg',
            zhs2zht=True,
            remove_url=True,
            # 设置为True，输出 onehot array
            to_onehot_array=True,
        )

        return word_feature_encoder,seg_feature_encoder

    @staticmethod
    def cross_validation(cv_data, test_data, result_file_path, **kwargs):
        """
            进行参数的交叉验证

        :param cv_data: k份训练数据
        :type cv_data: array-like
        :param test_data: 测试数据
        :type test_data: array-like
        :return:
        """

        nb_epoch = kwargs['nb_epoch']
        verbose = kwargs['verbose']
        num_labels = kwargs['num_labels']
        word_input_length, seg_input_length = 10, 7
        remove_stopword = kwargs['remove_stopword']
        word2vec_to_solve_oov = kwargs['word2vec_to_solve_oov']
        rand_seed = kwargs['rand_seed']
        l1_conv_filter_type = kwargs['l1_conv_filter_type']
        l2_conv_filter_type = kwargs['l2_conv_filter_type']
        k = kwargs['k']
        lr = kwargs['lr']

        use_layer = kwargs['use_layer']

        layer1 = kwargs['layer1'] if kwargs.get('layer1', []) !=[] else [-1]
        layer2 = kwargs['layer2'] if kwargs.get('layer2', []) !=[] else [-1]
        hidden1 = kwargs['hidden1'] if kwargs.get('hidden1', []) !=[] else [-1]
        hidden2 = kwargs['hidden2'] if kwargs.get('hidden2', []) !=[] else [-1]

        # 详细结果保存到...
        detail_result_file_path = result_file_path
        fout = open(detail_result_file_path, 'w')
        print('=' * 150)
        print('调节的参数....')
        print('use_layer:%s'%use_layer)
        print('layer1:%s' % str(layer1))
        print('layer2:%s' % str(layer2))
        print('hidden1:%s' % str(hidden1))
        print('hidden2:%s' % str(hidden2))
        print('-' * 150)
        print('word_input_length:%d\nseg_input_length:%d' % (word_input_length, seg_input_length))
        print('使用word2vec:%s\nremove_stopword:%s\nnb_epoch:%d\nrand_seed:%d' % (
            word2vec_to_solve_oov, remove_stopword, nb_epoch, rand_seed))
        print('l1_conv_filter_type:%s' % l1_conv_filter_type)
        print('l2_conv_filter_type:%s' % l2_conv_filter_type)
        print('k:%s' % k)
        print('=' * 150)

        fout.write('=' * 150 + '\n')
        fout.write('cv结果:\n')
        fout.write('lr:%f\nnb_epoch:%d\nrand_seed:%d\n' % (lr,nb_epoch, rand_seed))
        fout.write('l1_conv_filter_type:%s\n' % l1_conv_filter_type)
        fout.write('l2_conv_filter_type:%s\n' % l2_conv_filter_type)
        fout.write('k:%s\n' % k)
        fout.write('=' * 150 + '\n')

        from data_processing_util.cross_validation_util import transform_cv_data,get_val_score
        word_feature_encoder,seg_feature_encoder = MultiChannelOnehotBowCNN.get_feature_encoder(
           ** {'word_input_length':word_input_length,
             'seg_input_length':seg_input_length}
        )


        all_cv_word_data = transform_cv_data(word_feature_encoder, cv_data, test_data, **kwargs)
        all_cv_seg_data = transform_cv_data(seg_feature_encoder, cv_data, test_data, **kwargs)
        cv_data = [([dev_word_X,dev_seg_X],dev_y,[val_word_X,val_seg_X],val_y,(word_feature_encoder,seg_feature_encoder)) for (dev_word_X, dev_y, val_word_X, val_y,word_feature_encoder),(dev_seg_X, dev_y, val_seg_X, val_y,seg_feature_encoder) in zip(all_cv_word_data,all_cv_seg_data)]

        # 交叉验证
        parmater = product(layer1, layer2, hidden1, hidden2)

        for l1,l2,h1,h2 in parmater:

            fout.write('=' * 150 + '\n')
            fout.write('layer1:%d,layer2:%d,hidden1:%d,hidden2:%d\n' % (l1, l2, h1, h2))
            print('layer1:%d,layer2:%d,hidden1:%d,hidden2:%d' % (l1,l2,h1,h2))

            l1_conv_filter =[]
            if 'conv1' in use_layer:
                l1_conv_filter.extend([
                    [l1, l1_conv_filter_type[0][0], -1, l1_conv_filter_type[0][1], (0, 1), 0., 'relu', 'none'],
                    [l1, l1_conv_filter_type[1][0], -1, l1_conv_filter_type[1][1], (0, 1), 0., 'relu', 'none'],
                    [l1, l1_conv_filter_type[2][0], -1, l1_conv_filter_type[2][1], (0, 1), 0., 'relu', 'none'],
                ])

            full_connected_layer_units = []

            if 'hidden1' in use_layer:
                full_connected_layer_units.append([h1, 0., 'relu', 'none'])

            parm = {'l1_conv_filter_type':l1_conv_filter,
                    'full_connected_layer_units':full_connected_layer_units,
                    'num_labels':num_labels,
                    'verbose':verbose,
                    'nb_epoch':nb_epoch,
                    'lr':lr
                    }
            get_val_score(MultiChannelOnehotBowCNN,cv_data,fout,**parm)



        fout.close()

    def print_model_descibe(self):
        import pprint
        detail = {'rand_seed': self.rand_seed,
                  'verbose': self.verbose,
                  'optimizers': self.optimizers,
                  'word_input_dim': self.word_input_dim,
                  'seg_input_dim': self.seg_input_dim,
                  'word_input_length': self.word_input_length,
                  'seg_input_length': self.seg_input_length,
                  'num_labels': self.num_labels,
                  'l1_conv_filter_type': self.l1_conv_filter_type,
                  'l2_conv_filter_type': self.l2_conv_filter_type,
                  'full_connected_layer_units':self.full_connected_layer_units,
                  'nb_epoch': self.nb_epoch,
                  'earlyStoping_patience': self.earlyStoping_patience,
                  'lr':self.lr,
                  'batch_size':self.batch_size,
                  }
        pprint.pprint(detail)
        logging.debug(detail)
        return detail

def test_onehot_bow_cnn():
    # 使用样例
    train_X = ['你好', '无聊', '测试句子', '今天天气不错', '我要买手机']
    trian_y = [1, 3, 2, 2, 3]
    test_X = ['句子', '你好', '你妹']
    test_y = [2, 3, 0]
    sentence_padding_length = 8
    from data_processing_util.feature_encoder.onehot_feature_encoder import FeatureEncoder
    word_feature_encoder = FeatureEncoder(
        sentence_padding_length=sentence_padding_length,
        verbose=0,
        need_segmented=True,
        full_mode=True,
        replace_number=True,
        remove_stopword=True,
        lowercase=True,
        padding_mode='left',
        add_unkown_word=True,
        feature_type='word',
        zhs2zht=True,
        remove_url=True,
        # 设置为True，输出 onehot array
        to_onehot_array=True,
    )

    train_X_word_feature = word_feature_encoder.fit_transform(train_X)
    test_X_word_feature = word_feature_encoder.transform(test_X)
    print(','.join(word_feature_encoder.vocabulary))
    print train_X_word_feature.shape
    print train_X_word_feature

    seg_feature_encoder = FeatureEncoder(
        sentence_padding_length=sentence_padding_length,
        verbose=0,
        need_segmented=True,
        full_mode=True,
        replace_number=True,
        remove_stopword=True,
        lowercase=True,
        padding_mode='left',
        add_unkown_word=True,
        feature_type='seg',
        zhs2zht=True,
        remove_url=True,
        # 设置为True，输出 onehot array
        to_onehot_array=True,
    )

    train_X_seg_feature = seg_feature_encoder.fit_transform(train_X)
    test_X_seg_feature = seg_feature_encoder.transform(test_X)
    print(','.join(seg_feature_encoder.vocabulary))
    print train_X_seg_feature.shape
    print train_X_seg_feature

    # quit()
    onehot_cnn = MultiChannelOnehotBowCNN(
        rand_seed=1377,
        verbose=1,
        feature_encoder=(word_feature_encoder,seg_feature_encoder),
        # optimizers='adadelta',
        optimizers='sgd',
        word_input_length=sentence_padding_length,
        seg_input_length=sentence_padding_length,
        word_input_dim=word_feature_encoder.vocabulary_size,
        seg_input_dim=seg_feature_encoder.vocabulary_size,
        num_labels=5,
        l1_conv_filter_type=[
            [1, 2, -1, 'valid', (0, 1), 0., 'relu', 'none'],
            [1, 3, -1, 'valid', (0, 1), 0., 'relu', 'none'],
            [1, -1, -1, 'bow', (0, 1), 0., 'relu', 'none'],
        ],
        l2_conv_filter_type=[
            # [16, 2, -1, 'valid',(2,1),0.5, 'relu', 'none']
        ],
        full_connected_layer_units=[
            (50, 0.5, 'relu', 'none'),
        ],
        embedding_dropout_rate=0.,
        nb_epoch=30,
        nb_batch=5,
        earlyStoping_patience=20,
        lr=1e-2,
    )
    onehot_cnn.print_model_descibe()
    # 训练模型
    # 从保存的pickle中加载模型
    # onehot_cnn.model_from_pickle('model/modelA.pkl')
    print(onehot_cnn.fit(([train_X_word_feature,train_X_seg_feature], trian_y),
                   ([test_X_word_feature,test_X_seg_feature], test_y)))
    print(trian_y)
    # loss, train_accuracy = onehot_cnn.model.evaluate(train_X_feature, trian_y)

    # onehot_cnn.accuracy((train_X_word_feature, trian_y), transform_input=False)
    print(onehot_cnn.batch_predict([test_X_word_feature,test_X_seg_feature], transform_input=False))
    print(onehot_cnn.batch_predict_bestn([test_X_word_feature,test_X_seg_feature], transform_input=False, bestn=2))
    quit()
    print onehot_cnn.batch_predict(test_X, transform_input=True)
    print onehot_cnn.predict(test_X[0], transform_input=True)
    onehot_cnn.accuracy((test_X, test_y), transform_input=True)
    # 保存模型
    # onehot_cnn.save_model('model/modelA.pkl')

    print onehot_cnn.predict('你好吗', transform_input=True)


if __name__ == '__main__':
    test_onehot_bow_cnn()
