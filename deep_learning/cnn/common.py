# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-06-23'; 'last updated date: 2017-01-10'
    Email:   '383287471@qq.com'
    Describe: CNN base class
                提供一些公共的函数
"""

import logging
import numpy as np
from sklearn.metrics import f1_score
import pickle
from base.common_model_class import CommonModel
import sys

__version__ = '1.1'
sys.setrecursionlimit(15000)


class CnnBaseClass(CommonModel):
    """
        CNN模型的公共父类,包含一些预定义的方法和变量。

        包含以下主要函数：
            1. build_model： 编译和构建模型
            2. create_network： 搭建真正的CNN网络
            3. create_multi_size_convolution_layer
            4. create_convolution_layer：创建卷积层，支持多size和单size。
            5. create_full_connected_layer: 创建多层全连接层。


            6. fit：拟合和训练模型
            7. transform：对输入进行转换
            8. to_categorical： 将 label/y转为 onehot编码

            9. batch_predict_bestn: 批量预测，输出前n个结果
            10. predict： 单句预测
            11. batch_predict： 批量预测
            12. get_layer_output: 获取CNN某一层的输出

            13. save_model：保存模型
            14. model_from_pickle：恢复模型

            15. accuracy：模型验证
            16. print_model_descibe：打印模型详情
    """
    __version__ = '1.1'

    def __init__(self,
                 rand_seed=1337,
                 verbose=0,
                 feature_encoder=None,
                 optimizers='sgd',
                 input_length=None,
                 num_labels=None,
                 # nb_epoch=100,
                 # earlyStoping_patience=50,
                 **kwargs
                 ):
        """
            1. 初始化参数，并检验参数合法性。
            2. 设置随机种子，构建模型

        :param rand_seed: 随机种子,假如设置为为None时,则随机取随机种子
        :type rand_seed: int
        :param verbose: 数值越大,输出更详细的信息
        :type verbose: int
        :param feature_encoder: 输入数据的设置选项，设置输入编码器
        :type feature_encoder: onehot_feature_encoder.FeatureEncoder
        :param optimizers: 求解的优化算法，目前支持: ['sgd','adadelta']
        :type optimizers: str
        :param input_length: cnn设置选项,输入句子(序列)的长度.
        :type input_length: int
        :param num_labels: cnn设置选项,最后输出层的大小,即分类类别的个数.
        :type num_labels: int
        :param nb_epoch: cnn设置选项,cnn迭代的次数.
        :type nb_epoch: int
        :param earlyStoping_patience: cnn设置选项,earlyStoping的设置,如果迭代次数超过这个耐心值,依旧不下降,则stop.
        :type earlyStoping_patience: int
        :param kwargs: 目前有 lr , batch_size ,show_validate_accuracy
        :type kwargs: dict
        """

        self.rand_seed = rand_seed
        self.verbose = verbose
        self.feature_encoder = feature_encoder
        self.optimizers = optimizers
        self.verbose = verbose
        self.input_length = input_length
        self.num_labels = num_labels
        self.kwargs = kwargs

        assert optimizers in ['sgd', 'adadelta'], 'optimizers只能取 sgd, adadelta！'

        # 优化设置选项
        # 1. 模型学习率
        self.lr = self.kwargs.get('lr', 1e-2)
        # 批量大小
        self.batch_size = self.kwargs.get('batch_size', 32)
        self.nb_epoch = self.kwargs.get('nb_epoch', 30)
        self.earlyStoping_patience = self.kwargs.get('earlyStoping_patience', 25)

        # cnn model
        self.model = None
        # cnn model early_stop
        self.early_stop = None

        # region 中间层输出
        self.middle_layer_output = None
        # 下面全部注释掉，只用middle_layer_output集合在一起的就行
        # # 第一层卷积层输出
        # self.conv1_feature_output = None
        # # 第二层卷积层输出
        # self.conv2_feature_output = None
        # # 最后一层隐含层（倒数第二层）的输出
        # self.last_hidden_layer = None
        # # 输出层的输出
        # self.output_layer = None
        # cnn model 的输出函数
        # self.model_output = None
        # endregion

        # 选定随机种子
        if self.rand_seed is not None:
            np.random.seed(self.rand_seed)
            # 构建模型
            # self.build_model()

    def create_multi_size_convolution_layer(self,
                                            input_layer=None,
                                            convolution_filter_type=None,
                                            **kwargs
                                            ):

        """
            创建一个多类型（size，大小）核卷积层模型，可以直接添加到 keras的模型中去。
                1. 为每种size的核分别创建 Sequential 模型，模型内 搭建一个 2D卷积层 和一个 k-max pooling层
                2. 将1步骤创建的卷积核的结果 进行 第1维的合并，变成并行的卷积核
                3. 返回一个 4D 的向量

        必须是一个4D的输入，(n_batch,channel,row,col)

        :param convolution_filter_type: 卷积层的类型.一种 size对应一个 list

            for example:每个列表代表一种类型(size)的卷积核,和 max pooling 的size
                - 每一维的分别对应：nb_filter, nb_row, nb_col, border_mode, k,dropout_rate。如果nb_col设置-1的话，则nb_col=input_shape[-1]
                -  注意：每一列的第一行（即nb_filter）都应该是一样的，不然会报错
                convolution_filter_type = [[100,2,word_embedding_dim,'valid',(1,1), 0.5, 'relu', 'batch_normalization'],
                                    [100,4,word_embedding_dim,'valid',(1,1), 0., 'relu', 'batch_normalization'],
                                    [100,6,word_embedding_dim,'valid',(1,1), 0., 'relu', 'batch_normalization'],
                                   ]
        :type convolution_filter_type: array-like
        :param input_shape: 输入的 shape，3D，类似一张图，(channel,row,col)比如 （1,5,5）表示单通道5*5的图片
        :type input_shape: array-like
        :param k: 设置 k-max 层 的 k
        :type k: int
        :return: convolution model，4D-array
        :rtype: Sequential
        """

        # assert len(input_shape) == 3, 'warning: 因为必须是一个4D的输入，(n_batch,channel,row,col)，所以input shape必须是一个3D-array，(channel,row,col)!'

        from keras.layers import Dropout, merge, BatchNormalization, Activation

        dropout_rate = convolution_filter_type[0][-3]
        activation = convolution_filter_type[0][-2]
        normalization = convolution_filter_type[0][-1]

        self.check_param(activation=activation, normalization=normalization)

        # 构建第一层卷积层和1-max pooling
        conv_layers = []
        for items in convolution_filter_type:
            nb_filter, nb_row, nb_col, border_mode, k, _, _, _ = items
            m = self.create_one_size_convolution_layer(
                input_layer,
                nb_filter,
                nb_row,
                nb_col,
                border_mode,
                k,
                dropout_rate=0,
                activation='none',
                normalization='none',
                **kwargs
            )
            # m.summary()

            conv_layers.append(m)

        # 卷积的结果进行拼接
        # cnn_model = Sequential(**kwargs)
        output = merge(conv_layers, mode='concat', concat_axis=2)

        # 增加一个 规范化层
        if normalization == 'batch_normalization':
            output = BatchNormalization(axis=1, mode=2)(output)
        elif normalization == 'none':
            pass
        else:
            raise NotImplementedError
        # 增加一个 激活函数
        if activation != 'none':
            output = Activation(activation)(output)

        if dropout_rate > 0:
            output = Dropout(p=dropout_rate)(output)

        # -------------- print start : just print info -------------
        # if self.verbose > 1 :
        #    cnn_model.summary()
        # -------------- print end : just print info -------------
        # print(cnn_model.get_output_shape_at(-1))
        return output

    def create_one_size_convolution_layer(
            self,
            input_layer,
            nb_filter,
            nb_row,
            nb_col,
            border_mode,
            k,
            dropout_rate,
            activation,
            normalization,
            **kwargs
    ):

        self.check_param(activation=activation, normalization=normalization)

        from keras.layers import Dropout, Activation, BatchNormalization, Reshape
        from custom_layers import Convolution2DWrapper, MaxPooling2DWrapper

        # 普通2D卷积
        conv_output = Convolution2DWrapper(
            nb_filter,
            nb_row,
            nb_col,
            border_mode=border_mode,
            **kwargs
        )(input_layer)

        # 增加一个max pooling层
        if k[0] != 0:
            # if k[0]==0,则关闭pooling
            output = MaxPooling2DWrapper(pool_size=k)(conv_output)
        else:
            output = conv_output

        # 增加一个 规范化层
        if normalization == 'batch_normalization':
            output = BatchNormalization(axis=1, mode=2)(output)
        elif normalization == 'none':
            pass
        else:
            raise NotImplementedError

        # 增加一个 激活函数
        if activation != 'none':
            output = Activation(activation)(output)

        if dropout_rate > 0:
            output = Dropout(p=dropout_rate)(output)

        return output

    def create_convolution_layer(
            self,
            input_layer=None,
            convolution_filter_type=None,
            **kwargs
    ):
        '''
            创建一个卷积层模型，在keras的Convolution2D基础进行封装，使得可以创建多size和多size的卷积层

        :param input_layer: 上一层
        :param convolution_filter_type: 卷积核类型，可以多size和单size，比如：
            - 每一维的分别对应：（num_conv，conv_row，conv_col，conv_type，(max-pooling size),dropout_rate，activation，normalization）

                1) num_conv:卷积核个数: 多size时，每一种（每个列表）的num_conv都应该是一样的，不然会报错
                2) conv_row: 卷积核行
                3) conv_col: 卷积核列,设置-1的话，则nb_col=input_shape[-1]
                4) conv_type:卷积类型
                5) max-pooling size: max pooling filter 的大小,(k[0],k[1]),
                    - k[0]<0的话，使用普通 max pooling，size为( abs(k[0]) , k[1] )
                    - k[0]>1,使用k-max pooling
                    - k[0]==1,使用 1-max pooling
                6) dropout_rate: dropout rate，设为0的时候关闭,当使用多size的时候，dropout rate 以第一个为准，其他无视。
                6) activation: 激活函数 ，['linear','relu'],当使用多size的时候，activation 以第一个为准，其他无视。
                7) normalization: 规范化，['none','batch_normalization']，设置none的时候不使用,,当使用多size的时候，normalization 以第一个为准，其他无视。

            1. 多size：每个列表代表一种类型(size)的卷积核,分别为
                l1_conv_filter_type = [[100,2,word_embedding_dim,'valid',(1,1), 0.5, 'relu', 'batch_normalization'],
                                    [100,4,word_embedding_dim,'valid',(1,1), 0., 'relu', 'batch_normalization'],
                                    [100,6,word_embedding_dim,'valid',(1,1), 0., 'relu', 'batch_normalization'],
                                   ]
            2. 单size：一个列表即可。[[100,2,1,'valid',(k,1), 0.5, 'relu', 'batch_normalization']]

        :return: kera TensorVariable,output
        '''

        if len(convolution_filter_type) == 0:
            # 卷积类型为空，则直接返回
            output = input_layer

        elif len(convolution_filter_type) == 1:
            # 单size 卷积层
            nb_filter, nb_row, nb_col, border_mode, k, dropout_rate, activation, normalization = \
                convolution_filter_type[0]

            output = self.create_one_size_convolution_layer(
                input_layer,
                nb_filter,
                nb_row,
                nb_col,
                border_mode,
                k,
                dropout_rate,
                activation,
                normalization,
                **kwargs
            )

            # output = output_layer(input_layer)
            # output_shape = output_layer.get_output_shape_at(-1)
            # output_layer.summary()
        else:
            # 多size 卷积层
            output = self.create_multi_size_convolution_layer(
                input_layer=input_layer,
                convolution_filter_type=convolution_filter_type,
                **kwargs
            )

        return output

    def check_param(self, **kwargs):
        '''
            检验参数的合法性,activation,normalization

        :param kwargs: activation,normalization
        :return:
        '''

        if kwargs.has_key('activation'):
            assert kwargs['activation'] in ['none', 'linear',
                                            'relu'], 'create convolution layer error!activation 仅支持 none,linear,relu'
        if kwargs.has_key('normalization'):
            assert kwargs['normalization'] in ['none',
                                               'batch_normalization'], 'create convolution layer error!normalization 仅支持 none,batch_normalization'

        if kwargs.has_key('regularizer'):
            assert kwargs['regularizer'] in ['l1', 'l2', 'l1l2', 'none'], 'regularizer 只能取 [l1,l2,l1l2,none]'

        if kwargs.has_key('constraints'):
            assert kwargs['constraints'] in ['maxnorm', 'none'], 'regularizer 只能取 [maxnorm,none]'

    def create_full_connected_layer(
            self,
            input_layer=None,
            units=None,
    ):
        """创建多层的全连接层

        Parameters
        ----------
        input_layer : object
            上一层
        units : array-like
            - 每一层全连接层的单元数，分别对应
            - 1. unit，
            - 2. dropout rate，
            - 3. activation，
            - 4. normalization，
            - 5. （regularizer，regularizer_value）)
            - 6. （constraints，constraints_value）)

        Returns
        ----------
        output: object

        Notes
        -----------
        - 假如units设置为0,则不使用隐含层
        - regularizer 只能取 [l1,l2,l1l2,none]
        - regularizer 只能取 [maxnorm,none]

        Examples
        ---------
        Create a units list:

        >>> units =[
        >>> [50,0.5,'relu','batch_normalization',('l2',0.1),('maxnorm',3)],
        >>> [100,0.25,'relu','none'),('l1',0.1),('maxnorm',3)],
        >>> [100,0.25,'relu','none'),('l1l2',0.1),('none',0)],
        >>> ]

        """

        import keras.backend as K
        from keras.layers import Dense, Dropout, BatchNormalization, Activation, Reshape, Flatten
        from keras.regularizers import l2, l1, l1l2
        from keras.constraints import maxnorm
        from custom_layers import TransposeLayer
        # output_layer = Sequential(name='full_connected_layer')
        output = input_layer
        if K.ndim(output) != 2:
            output = Flatten()(output)

        regularizer, regularizer_value = None, None
        constraints, constraints_value = None, None
        for index, unit in enumerate(units):

            if type(unit) == int:
                unit = list([unit])
            num_dense = unit[0]
            if len(unit) == 1:
                dropout_rate = 0.
                activation = 'linear'
                normalization = 'none'
            elif len(unit) == 2:
                dropout_rate = unit[1]
                activation = 'linear'
                normalization = 'none'
            elif len(unit) == 3:
                dropout_rate = unit[1]
                activation = unit[2]
                normalization = 'none'
            elif len(unit) == 4:
                dropout_rate = unit[1]
                activation = unit[2]
                normalization = unit[3]
            elif len(unit) == 5:
                dropout_rate = unit[1]
                activation = unit[2]
                normalization = unit[3]
                regularizer, regularizer_value = unit[4]
            elif len(unit) == 6:
                dropout_rate = unit[1]
                activation = unit[2]
                normalization = unit[3]
                regularizer, regularizer_value = unit[4]
                constraints, constraints_value = unit[5]
            else:
                raise NotImplementedError

            self.check_param(activation=activation, normalization=normalization, regularizer=regularizer)
            if num_dense == 0:
                # 假如 设置为0,则不经过隐含层
                pass
            else:
                # region 设置 regularizers 和 constraints
                if regularizer is not None:
                    if regularizer == 'l1':
                        W_regularizer = l1(regularizer_value)
                    elif regularizer == 'l2':
                        W_regularizer = l2(regularizer_value)
                    elif regularizer == 'l1l2':
                        assert len(regularizer_value) == 2, 'regularizer_value 应该有两维！'
                        W_regularizer = l1l2(regularizer_value[0], regularizer_value[1])
                    elif regularizer == 'none':
                        W_regularizer = None
                    else:
                        raise NotImplementedError
                else:
                    W_regularizer = None
                # 设置constraints
                if regularizer is not None:
                    if constraints == 'maxnorm':
                        W_constraint = maxnorm(constraints_value)
                    elif constraints == 'none':
                        W_constraint = None
                    else:
                        raise NotImplementedError
                else:
                    W_constraint = None

                output = Dense(output_dim=num_dense,
                               init="glorot_uniform",
                               W_regularizer=W_regularizer,
                               W_constraint=W_constraint,
                               )(output)
                # endregion
            # 增加一个 规范化层
            if normalization == 'batch_normalization':
                output = TransposeLayer(axis=(0, 'x', 1))(output)
                output = BatchNormalization(mode=2, axis=1)(output)
                output = Flatten()(output)
            elif normalization == 'none':
                pass
            else:
                raise NotImplementedError
            # 增加一个 激活函数
            if activation != 'none':
                output = Activation(activation)(output)

            if dropout_rate > 0:
                output = Dropout(dropout_rate, name='full_connected_dropout%d_%.2f' % (index, dropout_rate))(output)

        return output

    def build_model(self):
        '''
            1. 创建 CNN 网络
            2. 设置优化算法,earlystop等

        :return:
        '''

        from keras.optimizers import SGD, Adadelta
        from keras.callbacks import EarlyStopping

        # 1. 创建CNN网络
        self.model = self.create_network()

        # -------------- region start : 2. 设置优化算法,earlystop等 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('2. 设置优化算法,earlystop等')
            print('2. 设置优化算法,earlystop等')
        # -------------- code start : 开始 -------------

        if self.optimizers == 'sgd':
            optimizers = SGD(lr=self.lr, decay=1e-6, momentum=0.9, nesterov=True)
        elif self.optimizers == 'adadelta':
            # optimizers = 'adadelta'
            optimizers = Adadelta(lr=self.lr, rho=0.95, epsilon=1e-6)
        else:
            optimizers = 'adadelta'
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizers, metrics=['accuracy'])
        self.early_stop = EarlyStopping(patience=self.earlyStoping_patience, verbose=self.verbose)

        # -------------- code start : 结束 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
            # -------------- region end : 2. 设置优化算法,earlystop等 ---------------

    def create_network(self):
        '''
            1. 创建 CNN 网络

                1. 输入层，2D，（n_batch,input_length）
                2. Embedding层,3D,（n_batch,input_length,embedding_dim）
                3. 输入dropout层，对Embedding层进行dropout.3D.
                4. Reshape层： 将embedding转换4-dim的shape，4D
                5. 第一层多size卷积层（含1-max pooling），使用三种size.
                6. Flatten层： 卷积的结果进行拼接,变成一列隐含层
                7. output hidden层
                8. output Dropout层
                9. softmax 分类层
            2. compile模型

        :return: cnn model network
        '''

        print('这是common类的create_network函数，如需创建特定任务的模型，请覆盖该方法！')
        from keras.layers import Embedding, Input, Activation, Reshape, Dropout, Dense, Flatten
        from keras.models import Model
        from keras import backend as K

        # 1. 输入层
        model_input = Input((self.input_length,), dtype='int32')
        # 2. Embedding层
        if self.embedding_init_weight is None:
            weight = None
        else:
            weight = [self.embedding_init_weight]
        embedding = Embedding(input_dim=self.input_dim,
                              output_dim=self.word_embedding_dim,
                              input_length=self.input_length,
                              # mask_zero = True,
                              weights=weight,
                              init='uniform'
                              )(model_input)
        # 3. Dropout层，对Embedding层进行dropout
        # 输入dropout层,embedding_dropout_rate!=0,则对embedding增加doupout层
        if self.embedding_dropout_rate:
            embedding = Dropout(p=self.embedding_dropout_rate)(embedding)

        # 4. Reshape层： 将embedding转换4-dim的shape
        embedding_4_dim = Reshape((1, self.input_length, self.word_embedding_dim))(embedding)
        # 5. 第一层多size卷积层（含1-max pooling），使用三种size.
        cnn_model = self.create_multi_size_convolution_layer(
            input_shape=(1,
                         self.input_length,
                         self.word_embedding_dim
                         ),
            convolution_filter_type=self.conv_filter_type,
        )
        # cnn_model.summary()
        conv1_output = cnn_model([embedding_4_dim] * len(self.conv_filter_type))
        # 6. Flatten层： 卷积的结果进行拼接,变成一列隐含层
        l6_flatten = Flatten()(conv1_output)
        # 7. output hidden层
        full_connected_layers = Dense(output_dim=self.num_labels, init="glorot_uniform", activation='relu')(l6_flatten)
        # 8. output Dropout层
        dropout_layers = Dropout(p=self.output_dropout_rate)(full_connected_layers)
        # 9. softmax 分类层
        softmax_output = Activation("softmax")(dropout_layers)
        model = Model(input_layer=[model_input], output=[softmax_output])

        # softmax层的输出
        self.model_output = K.function([model_input, K.learning_phase()], [softmax_output])
        # 卷积层的输出，可以作为深度特征
        self.conv1_feature_output = K.function([model_input, K.learning_phase()], [l6_flatten])
        if self.verbose > 1:
            model.summary()

        return model

    def to_categorical(self, y):
        '''
        将y转成适合CNN的格式,即标签y展开成onehot编码,比如
            y = [1,2]--> y = [[0,1 ],[1,0]]
        :param y: 标签列表,比如: [1,1,2,3]
        :type y: array1D-like
        :return: y的onehot编码
        :rtype: array2D-like
        '''
        from keras.utils import np_utils
        y_onehot = np_utils.to_categorical(y, nb_classes=self.num_labels)
        # quit()
        return y_onehot

    def transform(self, data):
        '''
            批量转换数据转换数据

        :param data: array-like,2D
        :return: feature
        '''

        feature = self.feature_encoder.transform(data)

        return feature

    def fit(self, train_data=None, validation_data=None):
        '''
            cnn model 的训练
                1. 对数据进行格式转换,比如 转换 y 的格式:转成onehot编码
                2. 模型训练

        :param train_data: 训练数据,格式为:(train_X, train_y),如果模型要求输入多个输入，则train_X中可用列表的方式存放多个输入即可,train_y是一个整形列表.
        :type train_data: (array-like,array-like)
        :param validation_data: 验证数据,格式为:(validation_X, validation_y),如果模型要求输入多个输入，则validation_X中可用列表的方式存放多个输入即可,validation_y是一个整形列表.
        :type validation_data: (array-like,array-like)
        :return: train_loss,train_accuracy,val_loss,val_accuracy
        '''
        # -------------- region start : 1. 对数据进行格式转换,比如 转换 y 的格式:转成onehot编码 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('2. 对数据进行格式转换,比如 转换 y 的格式:转成onehot编码')
            print('2. 对数据进行格式转换,比如 转换 y 的格式:转成onehot编码')
        # -------------- code start : 开始 -------------


        train_X, train_y = train_data
        validation_X, validation_y = validation_data

        train_y = self.to_categorical(train_y)
        validation_y = self.to_categorical(validation_y)
        # print(train_X.shape)
        # -------------- code start : 结束 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
        # -------------- region end : 1. 对数据进行格式转换,比如 转换 y 的格式:转成onehot编码 ---------------

        # -------------- region start : 2. 模型训练 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('3. 模型训练')
            print('3. 模型训练')
        # -------------- code start : 开始 -------------
        self.model.fit(
            train_X,
            train_y,
            nb_epoch=self.nb_epoch,
            verbose=self.verbose,
            # validation_split=0.1,
            validation_data=(validation_X, validation_y) if self.kwargs.get('show_validate_accuracy', True) else None,
            shuffle=True,
            batch_size=self.batch_size,
            callbacks=[self.early_stop]
        )

        train_loss, train_accuracy = self.model.evaluate(train_X, train_y, verbose=0)
        val_loss, val_accuracy = self.model.evaluate(validation_X, validation_y, verbose=0)

        # -------------- code start : 结束 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
        # -------------- region end : 2. 模型训练 ---------------
        return train_loss, train_accuracy, val_loss, val_accuracy

    def save_model(self, path):
        '''
            保存模型,保存成pickle形式
        :param path: 模型保存的路径
        :type path: 模型保存的路径
        :return:
        '''

        model_file = open(path, 'wb')
        pickle.dump(self.feature_encoder, model_file)
        pickle.dump(self.model, model_file)

    def model_from_pickle(self, path):
        '''
            从模型文件中直接加载模型
        :param path:
        :return: RandEmbeddingCNN object
        '''

        model_file = file(path, 'rb')
        self.feature_encoder = pickle.load(model_file)
        self.model = pickle.load(model_file)

    def predict(self, sentence, transform_input=False):
        '''
            预测一个句子的类别,对输入的句子进行预测 best1

        :param sentence: 测试句子,原始字符串句子即可，内部已实现转换成字典索引的形式
        :type sentence: str
        :param transform_input: 是否转换句子，如果为True,输入原始字符串句子即可，内部已实现转换成字典索引的形式。
        :type transform_input: bool
        '''

        y_pred = self.batch_predict([sentence], transform_input)[0]

        return y_pred

    def batch_predict_bestn(self, sentences, transform_input=False, bestn=1):
        '''
            批量预测句子的类别,对输入的句子进行预测

        :param sentences: 测试句子,
        :type sentences: array-like
        :param transform_input: 是否转换句子，如果为True,输入原始字符串句子即可，内部已实现转换成字典索引的形式。
        :type transform_input: bool
        :param bestn: 预测，并取出bestn个结果。
        :type bestn: int
        '''
        if transform_input:
            sentences = self.transform(sentences)
        # sentences = np.asarray(sentences)
        # assert len(sentences.shape) == 2, 'shape必须是2维的！'

        y_pred_prob = self.model.predict(sentences)
        y_pred_result = y_pred_prob.argsort(axis=-1)[:, ::-1][:, :bestn]
        y_pred_score = np.asarray([score[index] for score, index in zip(y_pred_prob, y_pred_result)])

        return y_pred_result, y_pred_score

    def batch_predict(self, sentences, transform_input=False):
        '''
            批量预测句子的类别,对输入的句子进行预测,只是输出 best1的将诶过

        :param sentences: 测试句子,
        :type sentences: array-like
        :param transform_input: 是否转换句子，如果为True,输入原始字符串句子即可，内部已实现转换成字典索引的形式。
        :type transform_input: bool
        '''

        y_pred, _ = self.batch_predict_bestn(sentences, transform_input, 1)
        y_pred = y_pred.flatten()

        return y_pred

    def get_layer_output(self, sentence, layer='hidden2', transform_input=False):
        """
            获取某一层的输出

        :param sentence: 测试句子,['','']
        :type sentence: array-like
        :param layer: 指定某一层的神经网络的输出, 'conv1','conv2','hidden1','hidden2'
        :type layer: str
        :param transform: 是否转换句子，如果为True,输入原始字符串句子即可，内部已实现转换成字典索引的形式。
        :type transform: array-like
        """

        assert layer in ['conv1', 'conv2', 'hidden1', 'hidden2',
                         'output'], 'layer 仅支持 conv1,conv2,hidden1,hidden2,output'

        assert self.middle_layer_output is not None, '功能没开启，请先设置 save_middle_output=True'

        if transform_input:
            assert type(sentence) == list, 'sentence 的 type 为 list！'
            sentence = self.transform(sentence)

        output = self.middle_layer_output([sentence, 0])

        # -------------- print start : just print info -------------
        if self.verbose > 2:
            logging.debug('句子表示成%d维的特征' % (output.shape))
            print('句子表示成%d维的特征' % (len(output)))

        # -------------- print end : just print info -------------
        return output

    def accuracy(self, test_data, transform_input=False):
        """
            预测,对输入的句子进行预测,并给出准确率
                1. 转换格式
                2. 批量预测
                3. 统计准确率等
                4. 统计F1(macro) :统计各个类别的F1值，然后进行平均

        :param sentence_index: 测试句子,以字典索引的形式
        :type sentence_index: array-like
        :return: y_pred,is_correct,accu,f1,test_loss
        :rtype:tuple
        """

        # -------------- region start : 1. 转换格式 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('1. 转换格式')
            print('1. 转换格式')
        # -------------- code start : 开始 -------------

        test_X, test_y = test_data
        if transform_input:
            test_X = self.transform(test_X)
        # test_X = np.asarray(test_X)

        # -------------- code start : 结束 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
        # -------------- region end : 1. 转换格式 ---------------

        # -------------- region start : 2. 批量预测 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('2. 批量预测')
            print('2. 批量预测')
        # -------------- code start : 开始 -------------

        y_pred = self.batch_predict(test_X)
        test_loss, test_accuracy = self.model.evaluate(
            test_X,
            self.to_categorical(test_y),
            verbose=0,
        )

        # -------------- code start : 结束 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
        # -------------- region end : 2. 批量预测 ---------------

        # -------------- region start : 3 & 4. 计算准确率和F1值 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('3 & 4. 计算准确率和F1值')
            print('3 & 4. 计算准确率和F1值')
        # -------------- code start : 开始 -------------

        is_correct = y_pred == test_y
        logging.debug('正确的个数:%d' % (sum(is_correct)))
        print('正确的个数:%d' % (sum(is_correct)))
        accu = sum(is_correct) / (1.0 * len(test_y))
        logging.debug('准确率为:%f' % (accu))
        print('准确率为:%f' % (accu))
        print('测试误差为：%f' % test_loss)
        f1 = f1_score(test_y, y_pred.tolist(), average=None)
        logging.debug('F1为：%s' % (str(f1)))
        print('F1为：%s' % (str(f1)))

        # -------------- code start : 结束 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
        # -------------- region end : 3 & 4. 计算准确率和F1值 ---------------

        return y_pred, is_correct, accu, f1, test_loss

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
                  }
        pprint.pprint(detail)
        logging.debug(detail)
        return detail
