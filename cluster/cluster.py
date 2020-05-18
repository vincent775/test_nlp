# -*-Coding: UTF-8 -*-

#使用K-means 对豆瓣读书数据聚类


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#读取文件
book_data = pd.read_csv('data/data.csv')
print(book_data.head())
book_titles = book_data['title'].tolist()
book_content = book_data['content'].tolist()
print('书名：',book_titles[0])
print('内容：',book_content[0])

#提取标签入口
def build_feature_matrix(documents,feature_type='frequency',ngram_range=(1,1),min_df=0.0,max_df=1.0):
    feature_type = feature_type.lower().strip()
    if feature_type=="binary":
        vectorizer =  CountVectorizer(binary=True,
                                      max_df=max_df,ngram_range=ngram_range)
    elif feature_type=='frequency':
        vectorizer = CountVectorizer(binary=False,min_df=min_df,
                                     max_df=max_df,ngram_range=ngram_range)
    elif feature_type=='tfidf':
        vectorizer = TfidfVectorizer()
    else:
        raise Exception("Wrong feature type entered.Possible values:'binary','frequency','tfidf'")
    feature_matrix = vectorizer.fit_transform(documents).astype(float)
    return vectorizer,feature_matrix

#加载数据处理文件
from normalization import normalize_corpus

norm_book_content = normalize_corpus(book_content)



#提取tf-idf 特征
vectorizer,feature_matrix = build_feature_matrix(norm_book_content,
                                                 feature_type='tfidf',
                                                 min_df=0.2,max_df=0.90,
                                                 ngram_range=(1,2))

# 查看特征数量
print('特征数量：',feature_matrix.shape)
#获取特征名字
feature_names = vectorizer.get_feature_names()
#打印某些特征
print('抽取个别特征：',feature_names[:10])

#提取完特征之后进行聚类
from sklearn.cluster import KMeans
def k_means(feature_matrix,num_clusters=10):
    km = KMeans(n_clusters=num_clusters,
                max_iter=10000)
    km.fit(feature_matrix)
    clusters = km.labels_
    return km,clusters

def get_cluster_data(clustering_obj,book_data,feature_names,num_clusters,topn_features=10):
    cluster_details = {}
    #获取cluster的center
    ordered_centroids =  clustering_obj.cluster_centers_.argsort()[:,::-1]
    #获得每个cluster的关键特征
    #获得每个cluster的书
    for cluster_num in range(num_clusters):
        cluster_details[cluster_num] = {}
        cluster_details[cluster_num]['cluster_num'] = cluster_num
        key_features = [feature_names[index] for index in ordered_centroids[cluster_num,:topn_features]]
        cluster_details[cluster_num]['key_features'] = key_features
        books = book_data[book_data['Cluster'] == cluster_num]['title'].values.tolist()
        cluster_details[cluster_num]['books'] = books
    return cluster_details

def print_cluster_data(cluster_data):
    # print cluster details
    for cluster_num, cluster_details in cluster_data.items():
        print('Cluster {} details:'.format(cluster_num))
        print('-' * 20)
        print('Key features:', cluster_details['key_features'])
        print('book in this cluster:')
        print(', '.join(cluster_details['books']))
        print('=' * 40)

num_clusters = 10
km_obj,clusters = k_means(feature_matrix=feature_matrix,num_clusters=num_clusters)
book_data['Cluster'] = clusters
# 获取每个cluster的数量
from collections import Counter
c = Counter(clusters)
print('打印每一个clusters的数量：',c.items())

#打印每一个cluster的书籍
cluster_data = get_cluster_data(clustering_obj=km_obj,
                                book_data = book_data,
                                feature_names = feature_names,
                                num_clusters=num_clusters,
                                topn_features=10)
print_cluster_data(cluster_data)




