# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-07-06'; 'last updated date: 2016-09-27'
    Email:   '383287471@qq.com'
    Describe:
    #########    RF（BOC/BOW） 模型    #########
    Bag of Words (BOW) or TFIDF as input features, random forest (RF) as classifier.


"""

from __future__ import print_function

import cPickle as pickle
import logging
import pprint
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from base.common_model_class import CommonModel
from data_processing_util.feature_encoder.bow_feature_encoder import FeatureEncoder


class BowRandomForest(CommonModel):
    """
        使用BOW或者TFIDF作为特征输入，使用random forest作为分类器
        函数有：
            1. transform
            2. fit
            3、cross_validation
            4、get_model ： get BowRandomForset object

        Examples
        --------
        >>> BowRandomForest.cross_validation(

        )

    """

    def __init__(self, rand_seed=1337, verbose=0, n_estimators=200, min_samples_leaf=2, feature_encoder=None, **kwargs):
        '''
            1. 初始化参数，并检验参数合法性。
            2. 设置随机种子，构建模型

        :param rand_seed: 随机种子,假如设置为为None时,则随机取随机种子
        :type rand_seed: int
        :param verbose: 数值越大,输出更详细的信息
        :type verbose: int
        :param rand_seed: 随机种子,假如设置为为None时,则随机取随机种子
        :type rand_seed: int
        :param verbose: 数值越大,输出更详细的信息
        :type verbose: int
        :param feature_encoder: 输入数据的设置选项，设置输入编码器
        :type feature_encoder: bow_feature_encoder.FeatureEncoder
        :param kwargs: 目前有 ‘word2vec_model_file_path’
        '''

        super(BowRandomForest, self).__init__(rand_seed, verbose)

        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.feature_encoder = feature_encoder
        self.kwargs = kwargs

        # random forest model
        self.model = None

    def transform(self, data):
        '''
            批量转换数据转换数据

        :param train_data: array-like,1D
        :return:features
        '''

        features = self.feature_encoder.transform(data)
        return features

    def model_from_pickle(self, path):
        '''
            从模型文件中直接加载模型
        :param path:
        :return: RandEmbeddingCNN object
        '''
        fin = file(path, 'rb')
        self.model = pickle.load(fin)
        self.feature_encoder = pickle.load(fin)
        return self

    def save_model(self, path):
        '''
            保存模型,保存成pickle形式
        :param path: 模型保存的路径
        :type path: 模型保存的路径
        :return:
        '''
        fout = open(path, 'wb')
        pickle.dump(self.model, fout)
        pickle.dump(self.feature_encoder, fout)

    def batch_predict_bestn(self, sentences, transform_input=False, bestn=1):
        '''
            批量预测句子的类别,对输入的句子进行预测

        :param sentences: 测试句子,
        :type sentences: array-like
        :param transform: 是否转换句子，如果为True,输入原始字符串句子即可，内部已实现转换。
        :type transform: bool
        :param bestn: 预测，并取出bestn个结果。
        :type bestn: int
        '''
        if transform_input:
            sentences = self.transform(sentences)
        sentences = np.asarray(sentences)
        # print(sentences)
        assert len(sentences.shape) == 2, 'shape必须是2维的！'

        y_pred_prob = self.model.predict_proba(sentences)
        y_pred_result = y_pred_prob.argsort(axis=-1)[:, ::-1][:, :bestn]
        y_pred_score = np.asarray([score[index] for score, index in zip(y_pred_prob, y_pred_result)])
        return y_pred_result, y_pred_score

    def batch_predict(self, sentences, transform_input=False):
        '''
            批量预测句子的类别,对输入的句子进行预测

        :param sentences: 测试句子,
        :type sentences: array-like
        :param transform: 是否转换句子，如果为True,输入原始字符串句子即可，内部已实现转换。
        :type transform: bool
        '''
        y_pred, _ = self.batch_predict_bestn(sentences, transform_input, 1)
        y_pred = np.asarray(y_pred).flatten()

        return y_pred

    def fit(self, train_data=None, validation_data=None):
        '''
            rf model 的训练
                1. 对数据进行格式转换,
                2. 模型训练

        :param train_data: 训练数据,格式为:(train_X, train_y),train_X中每个句子以字典索引的形式表示(使用data_processing_util.feature_encoder.onehot_feature_encoder编码器编码),train_y是一个整形列表.
        :type train_data: (array-like,array-like)
        :param validation_data: 验证数据,格式为:(validation_X, validation_y),validation_X中每个句子以字典索引的形式表示(使用data_processing_util.feature_encoder.onehot_feature_encoder编码器编码),validation_y是一个整形列表.
        :type validation_data: (array-like,array-like)
        :return: None
        '''
        # -------------- region start : 1. 对数据进行格式转换 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('1. 对数据进行格式转换')
            print('1. 对数据进行格式转换')
        # -------------- code start : 开始 -------------


        train_X, train_y = train_data
        train_X = np.asarray(train_X)
        validation_X, validation_y = validation_data
        validation_X = np.asarray(validation_X)

        # -------------- code start : 结束 -------------
        # -------------- region start : 训练模型 -------------
        if self.verbose > 2:
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('训练模型')
            print('训练模型')
        # -------------- code start : 开始 -------------
        # Initialize a Random Forest classifier with n trees
        forest = RandomForestClassifier(n_estimators=self.n_estimators,
                                        # min_samples_leaf = min_samples_leaf,
                                        # oob_score = True,
                                        max_features='log2',
                                        random_state=self.rand_seed)
        forest.fit(train_X, train_y)
        self.model = forest

        y_pred, is_correct, dev_accuracy, f1 = self.accuracy((train_X, train_y), transform_input=False)
        y_pred, is_correct, val_accuracy, f1 = self.accuracy((validation_X, validation_y), transform_input=False)

        dev_loss, val_loss = 0, 0

        # -------------- code start : 结束 -------------
        if self.verbose > 2:
            logging.debug('-' * 20)
            print('-' * 20)
        # -------------- region end : 训练模型 ---------------
        return dev_loss, dev_accuracy, val_loss, val_accuracy

    def predict(self, sentence, transform_input=False):
        '''
            预测一个句子的类别,对输入的句子进行预测

        :param sentences: 测试句子,
        :type sentences: str
        :param transform: 是否转换句子，如果为True,输入原始字符串句子即可，内部已实现转换成字典索引的形式。
        :type transform: bool
        '''
        y_pred = self.batch_predict([sentence], transform_input)[0]
        return y_pred

    def print_model_descibe(self):
        import pprint
        detail = {'rand_seed': self.rand_seed,
                  'verbose': self.verbose,
                  'n_estimators': self.n_estimators,
                  'min_samples_leaf': self.min_samples_leaf,
                  'vocabulary_size': self.feature_encoder.vocabulary_size,
                  'train_data_count': len(self.feature_encoder.train_data),
                  }
        pprint.pprint(detail)
        logging.debug(detail)
        return detail

    def accuracy(self, test_data, transform_input=False):
        '''
            预测,对输入的句子进行预测,并给出准确率
                1. 转换格式
                2. 批量预测
                3. 统计准确率等
                4. 统计F1(macro) :统计各个类别的F1值，然后进行平均

        :param test_data: 测试句子
        :type test_data: array-like
        :param transform_input:
        :type transform_input: bool
        :return: y_pred,is_correct,accu,f1
        :rtype:tuple
        '''
        # -------------- region start : 1. 转换格式 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
            logging.debug('1. 转换格式')
            print('1. 转换格式')
        # -------------- code start : 开始 -------------

        test_X, test_y = test_data

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

        y_pred = self.batch_predict(test_X, transform_input)

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
        accu = sum(is_correct) / (1.0 * len(test_y))
        # quit()
        f1 = f1_score(test_y, y_pred.tolist(), average=None)

        if self.verbose > 0:
            logging.debug('正确的个数:%d' % (sum(is_correct)))
            print('正确的个数:%d' % (sum(is_correct)))
            logging.debug('准确率为:%f' % (accu))
            print('准确率为:%f' % (accu))
            logging.debug('F1为：%s' % (str(f1)))
            print('F1为：%s' % (str(f1)))

        # -------------- code start : 结束 -------------
        if self.verbose > 1:
            logging.debug('-' * 20)
            print('-' * 20)
        # -------------- region end : 3 & 4. 计算准确率和F1值 ---------------

        return y_pred, is_correct, accu, f1

    @staticmethod
    def get_model(
            feature_encoder,
            n_estimators,
            **kwargs
    ):
        bow_randomforest = BowRandomForest(
            rand_seed=1377,
            verbose=kwargs.get('verbose', 0),
            feature_encoder=feature_encoder,
            # optimizers='adadelta',
            n_estimators=n_estimators,
            min_samples_leaf=1,
        )

        return bow_randomforest

    @staticmethod
    def get_feature_encoder(**kwargs):
        '''
            返回 该模型的输入 特征编码器

        :param kwargs: 可设置参数 [ full_mode(#,False), feature_type(#,word),verbose(#,0)],word2vec_to_solve_oov[#,False],word2vec_model_file_path[#,None],加*表示必须提供，加#表示可选，不写则默认。

        :return:
        '''

        feature_encoder = FeatureEncoder(
            verbose=kwargs.get('verbose', 0),
            need_segmented=kwargs.get('need_segmented', True),
            full_mode=kwargs.get('full_mode', False),
            replace_number=True,
            remove_stopword=True,
            lowercase=True,
            add_unkown_word=True,
            feature_type=kwargs.get('feature_type', 'word'),
            zhs2zht=True,
            remove_url=True,
            feature_method='bow',
            max_features=2000,
            word2vec_to_solve_oov=kwargs.get('word2vec_to_solve_oov', False),
            word2vec_model_file_path=kwargs.get('word2vec_model_file_path', None)
        )
        if kwargs.get('verbose', 0) > 0:
            pprint.pprint(kwargs)

        return feature_encoder

    @staticmethod
    def cross_validation(
            train_data=None,
            test_data=None,
            cv_data=None,
            shuffle_data=True,
            n_estimators_list=None,
            feature_type='word',
            word2vec_to_solve_oov=False,
            word2vec_model_file_path=None,
            verbose=0,
            cv=3,
            need_segmented=True,
            need_validation=True,
            include_train_data=True,
    ):
        """进行参数的交叉验证

        Parameters
        ----------
        train_data : (array-like,array-like)
            训练数据 (train_X,train_y)
        test_data : (array-like,array-like)
            测试数据 (test_X,test_y)
        cv_data : array-like
            k份验证数据
        word2vec_to_solve_oov : bool
            是否使用 w2v 去替换
        n_estimators_list : array-like
            验证参数，随机森林棵树
        feature_type : str
            特征类型, only in ['word','seg','word_seg']
        shuffle_data : bool
            是否打乱数据
        verbose : int
            数值越大，输出越详细
        cv:int
            进行 cv 折验证
        need_segmented:bool
            是否需要分词
        include_train_data:
            是否包含训练数据一样验证
        need_validation:
            是否要验证
        """

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
        feature_encoder = BowRandomForest.get_feature_encoder(
            verbose=verbose,
            need_segmented=need_segmented,
            feature_type=feature_type,
            word2vec_to_solve_oov=word2vec_to_solve_oov,
            word2vec_model_file_path=word2vec_model_file_path,
        )
        # diff_train_val_feature_encoder=1 每次feature encoder 都不同
        cv_data = transform_cv_data(feature_encoder, cv_data, verbose=verbose, diff_train_val_feature_encoder=1)

        # 交叉验证
        for n_estimators in n_estimators_list:
            print('=' * 40)
            print('n_estimators is %d.' % n_estimators)
            get_val_score(BowRandomForest,
                          cv_data=cv_data[:],
                          verbose=verbose,
                          shuffle_data=shuffle_data,
                          need_validation=need_validation,
                          n_estimators=n_estimators,
                          )



if __name__ == '__main__':
    train_X = ['你好', '无聊', '测试句子', '今天天气不错', '我要买手机']
    trian_y = [1, 3, 2, 2, 3]
    test_X = ['你妹', '句子', '你好']
    test_y = [2, 3, 0]

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
        max_features=2000,
        feature_type='seg',
    )
    train_X_feature = feature_encoder.fit_transform(train_X)
    test_X_feature = feature_encoder.transform(test_X)
    print(','.join(feature_encoder.vocabulary))

    print(train_X_feature)

    bow_rf = BowRandomForest(
        rand_seed=1337,
        verbose=0,
        n_estimators=200,
        min_samples_leaf=1,
        feature_encoder=feature_encoder,
        word2vec_to_solve_oov=True,
        word2vec_model_file_path='/home/jdwang/PycharmProjects/corprocessor/word2vec/vector/vector1000000_50dim.gem',
    )
    # bow_rf.model_from_pickle('model.pkl')


    bow_rf.fit(train_data=(train_X_feature, trian_y),
               validation_data=(test_X_feature, test_y))
    print(bow_rf.batch_predict(test_X, transform_input=True))
    print(bow_rf.predict('你好', transform_input=True))
    bow_rf.accuracy((train_X_feature, trian_y), False)
    # bow_rf.accuracy((test_X_feature,test_y),False)
    bow_rf.print_model_descibe()

    bow_rf.save_model('model.pkl')
