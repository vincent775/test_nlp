# -*-coding: UTF-8 -*-

#两种模式的特征提取方式
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
'''
  文本特征提取：
      将文本数据转化成特征向量的过程
      比较常用的文本特征表示法为词袋法
  词袋法：
      不考虑词语出现的顺序，每个出现过的词汇单独作为一列特征
      这些不重复的特征词汇集合为词表
      每一个文本都可以在很长的词表上统计出一个很多列的特征向量
      如果每个文本都出现的词汇，一般被标记为 停用词 不计入特征向量

  主要有两个api来实现 CountVectorizer 和 TfidfVectorizer
  CountVectorizer：
      只考虑词汇在文本中出现的频率
  TfidfVectorizer：
      除了考量某词汇在文本出现的频率，还关注包含这个词汇的所有文本的数量
      能够削减高频没有意义的词汇出现带来的影响, 挖掘更有意义的特征

  相比之下，文本条目越多，Tfid的效果会越显著


  下面对两种提取特征的方法，分别设置停用词和不停用，
  使用朴素贝叶斯进行分类预测，比较评估效果

  '''


#基于词袋模型提取特征
def bow_extractor(corpus,ngram_rang=(1,1)):
    vectorizer = CountVectorizer(min_df=1,ngram_range=ngram_rang)
    features =  vectorizer.fit_transform(corpus)
    return vectorizer,features

from  sklearn.feature_extraction.text import TfidfTransformer
#基于TF-Idf特征提取
def tfidf_transformer(bow_matrix):
    transformer = TfidfTransformer(norm='12',smooth_idf=True,use_idf=True)
    tfidf_matrix = transformer.fit_transform(bow_matrix)
    return transformer,tfidf_matrix


def tfidf_extractor(corpus,ngram_range=(1,1)):
    vectorizer = TfidfVectorizer(min_df=1,norm='l2',smooth_idf=True,use_idf=True,ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer,features


