# nlp_util
###NLP常用工具

### backup some common nlp util

###环境:
    - Ubuntu 14.04 / Linux mint 17.03
    - Python: 2.7.6版本.
    - python lib: 
        - jieba 0.38
        - Keras 1.0.4

##工具列表

###data_processing_util: 数据预处理类工具

- ####[jiebanlp:](https://github.com/JDwangmo/nlp_util/tree/master/data_processing_util/jiebanlp)
    - describe: jieba分词
    - 依赖包: jieba 0.38
    - 项目结构:
        - stopword.txt: 中文停止词
        - userdict.txt: 用户自定义字典
        

    
    
---------------
###deep_learning: 深度学习类工具

- ####[randEmbedding_cnn:](https://github.com/JDwangmo/nlp_util/tree/master/deep_learning/cnn/randEmbedding_cnn)
    - describe: CNN-rand.随机词向量输入,一层CNN,多种卷积核,具体见:[Kim,Convolutional Neural-Networks for Sentence Classification,2014](https://github.com/JDwangmo/coprocessor#2convolutional-neural-networks-for-sentence-classification).
    