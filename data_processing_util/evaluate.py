# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2017-04-07'; 'last updated date: 2017-04-07'
    Email:   '383287471@qq.com'
    Describe:
"""
from __future__ import print_function
from sklearn.metrics import f1_score
import numpy as np

__version__ = '1.0'


def get_evaluate_score(y_true, y_predict):
    accuracy = np.mean(y_true == y_predict)
    print('准确度:%f' % accuracy)
    print('错误个数: %d (总:%d)' % (sum(y_true != y_predict), len(y_true)))
    f1 = f1_score(y_true, y_predict, average=None)
    print('F1 score:', f1)
    return accuracy
