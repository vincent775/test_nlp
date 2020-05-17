# -*-coding: UTF-8 -*-

#两种模式的特征提取方式
from sklearn.feature_extraction.text import CountVectorizer

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
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_extractor(corpus,ngram_range=(1,1)):
    vectorizer = TfidfVectorizer(min_df=1,norm='12',smooth_idf=True,use_idf=True,ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer,features


