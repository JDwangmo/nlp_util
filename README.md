# nlp_util
###NLP常用工具

### summary:
1. 预处理类工具:分词
2. 深度学习:
    - cnn
3. 传统分类器:
    - random forest

### 环境:
- Ubuntu 14.04 / Linux mint 17.03
- Python: 2.7.6版本.
- python lib: 
    - jieba 0.38: 分词工具
        - 官网： https://github.com/fxsjy/jieba
    - Keras 1.0.4: 神经网络的框架
        - 官网： https://github.com/fchollet/keras
    - OpenCC 0.2: Open Chinese Convert 開放中文轉換
        - 官网： https://github.com/BYVoid/OpenCC 和 [Python 接口](https://github.com/lepture/opencc-python)
        - 安装方法：sudo pip install OpenCC
        
    - scikit-learn 0.17.1: 机器学习工具类，包括计算F1值等
        - 官网： https://github.com/scikit-learn/scikit-learn
        - 安装方法：sudo pip install scikit-learn


## 工具列表

### commom: 通用类
- common_model_class：分类器等模型的父类，规范分类器的函数等

---------------


### data_processing_util: 数据预处理类工具

- [jiebanlp:](https://github.com/JDwangmo/nlp_util/tree/master/data_processing_util/jiebanlp)
    - describe: jieba分词
    - 依赖包: jieba 0.38,OpenCC 0.2
    - 项目结构:
        - stopword.txt: 中文停止词表.
        - userdict.txt: 用户自定义字典
        - jieba_util.py: jieba分词工具类,自封装了一层,即包装成一个 Jieba_Util类,这个类主要在原 jieba 分词的基础上对扩展：
            - 增加用户字典
            - 是否转为小写
            - 是否移除stopwords，
            - 是否统一替换数字 
            - 是否繁体转简体
            
            
- [word2vec_util](https://github.com/JDwangmo/nlp_util/tree/master/data_processing_util/word2vec_util)      
    - word2vec模型的训练等
    - 依赖包：gensim 0.13.1,jieba 0.38
    
- [feature_encoder](https://github.com/JDwangmo/nlp_util/tree/master/data_processing_util/feature_encoder)      
    - 特征编码包，比如onehot编码，bow编码，tfidf编码等
    - 依赖包：gensim 0.13.1,jieba 0.38,scikit-learn 0.17.1
    - 项目结构：
        - `onehot_feature_encoder.py`:特征编码类,将原始输入的句子转换为补齐的字典索引的形式,使用0补长.
        - `bow_feature_encoder.py`:
    
    
---------------
### deep_learning: 深度学习类工具

#### [CNN类](https://github.com/JDwangmo/nlp_util/tree/master/deep_learning/cnn/)
- [cnn/wordEmbedding_cnn:]
    - describe: 
        - 基于CNN-rand,对输入层增加了dropout rate的调节,目的在于避免输入特征过多,训练时间长,另外可以设置k-max,不止原paper中模型的1-max.随机词向量输入或者pretraining词向量输入,一层CNN,多种卷积核,具体见:[Kim,Convolutional Neural-Networks for Sentence Classification,2014](https://github.com/JDwangmo/coprocessor#2convolutional-neural-networks-for-sentence-classification).
    
    - 依赖包: Keras 1.0.4, scikit-learn 0.17.1,
    - 项目结构:
        - [wordEmbedding_cnn_model.py]:
            - CNN-rand模型类,搭建一层卷积层的CNN-rand.集成了训练,测试,统计准确、F1值等方法.
            - 关于这个模型的具体内容请参考: [Kim et al.,Convolutional Neural-Networks for Sentence Classification,EMNLP 2014](https://github.com/JDwangmo/coprocessor#2convolutional-neural-networks-for-sentence-classification)
    
- [cnn/dynamic_cnn:]
    - describe:动态 k-max poooling 操作。
