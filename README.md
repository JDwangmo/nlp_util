# nlp_util
###NLP常用工具

### summary:
- 预处理类工具:分词
- 分类器:cnn

### 环境:
    - Ubuntu 14.04 / Linux mint 17.03
    - Python: 2.7.6版本.
    - python lib: 
        - jieba 0.38
        - Keras 1.0.4

## 工具列表

### data_processing_util: 数据预处理类工具

- [jiebanlp:](https://github.com/JDwangmo/nlp_util/tree/master/data_processing_util/jiebanlp)
    - describe: jieba分词
    - 依赖包: jieba 0.38
    - 项目结构:
        - stopword.txt: 中文停止词
        - userdict.txt: 用户自定义字典
        - jieba_util.py: jieba分词工具类,自封装了一层,内含 Jieba_Util类
        
    
    
---------------
### deep_learning: 深度学习类工具

#### [CNN类](https://github.com/JDwangmo/nlp_util/tree/master/deep_learning/cnn/)
- [cnn/randEmbedding_cnn:]
    - describe: 基于CNN-rand,对输入层增加了dropout rate的调节,目的在于避免输入特征过多,训练时间长,另外可以设置k-max,不止原paper中模型的1-max.随机词向量输入,一层CNN,多种卷积核,具体见:[Kim,Convolutional Neural-Networks for Sentence Classification,2014](https://github.com/JDwangmo/coprocessor#2convolutional-neural-networks-for-sentence-classification).
    - 依赖包: Keras 1.0.4
    - 项目结构:
        - [feature_encoder.py]:特征编码类,将原始输入的句子转换为补齐的字典索引的形式,使用0补长.
        - [randEmbedding_cnn_model.py]:CNN-rand模型类,搭建一层卷积层的CNN-rand.集成了训练,测试等方法.关于这个模型的具体内容请参考: [Kim et al.,Convolutional Neural-Networks for Sentence Classification,EMNLP 2014](https://github.com/JDwangmo/coprocessor#2convolutional-neural-networks-for-sentence-classification)
    