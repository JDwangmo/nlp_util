# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2017-02-25'; 'last updated date: 2017-02-25'
    Email:   '383287471@qq.com'
    Describe: 拼音排序
"""
import os
__version__ = '1.4'


def sort(words):
    """

    Parameters
    ----------
    words: array-like
        待排序的词汇列表

    Returns
    -------
    排完序的词汇列表: array-like

    """
    pinyin = file(os.path.dirname(__file__) + '/convert-utf-8.txt').read().split('\n')
    pinyin = [v for v in pinyin if v.strip() != ""]
    # print len(pinyin)
    one_list = []
    cnt1 = 0
    for cnt in xrange(len(pinyin)):
        try:
            tmp = pinyin[cnt].decode("gbk")
            one_list.append(tmp)
        except Exception, e:
            # print cnt
            cnt1 += 1
            continue
    # print len(one_list), cnt1
    one_list = [v.split(",")[0] for v in one_list]
    # print len(one_list)
    convert_dict = {}
    for v in one_list:
        convert_dict[v[0]] = v[1:]
    # print convert_dict[u"龥"]

    for v1 in words:
        for v2 in v1:
            if not convert_dict.has_key(v2):
                convert_dict[v2] = v2

    names_sort = sorted(words, key=lambda x: ''.join([convert_dict[v] for v in x]))
    return names_sort


if __name__ == '__main__':
    print(','.join(sort([u'你好', u'啊', u'好', u'苹果', u'大狗'])))
