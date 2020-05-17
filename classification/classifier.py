# -*- conding:UTF-8 -*-
#实现邮件的正常与垃圾的识别

import numpy as  np
from sklearn.model_selection import train_test_split

#数据提取
def get_data():
    '''
    获取诗句
    :return: 本文数据对应的 labels
    '''
    with open("data/ham_data.txt",encoding='utf-8') as ham_f,open('data/spam_data.txt',encoding='utf-8') as spam_f:
        ham_data = ham_f.readlines()
        spam_data = spam_f.readlines()
        ham_label = np.ones(len(ham_data)).tolist()
        spam_label = np.ones(len(spam_data)).tolist()
        corpus = ham_data+spam_data
        labels = ham_label+spam_label
    return corpus,labels

#区分训练集和测试集
def prepare_datasets(corpus,labels,test_data_proportion=0.3):
    '''
    :param corpus: 文本数据
    :param labels: label数据
    :param test_data_proportion: 测试数据集占比
    :return: 训练数据，测试数据，训练label，测试label
    '''
    #使用sklearn 框架进行拆分
    train_x,test_x,train_Y,test_Y = train_test_split(corpus,labels,test_size=test_data_proportion,random_state=42)
    return train_x,test_x,train_Y,test_Y

def remove_empty_docs(corpus, labels):
    filtered_corpus = []
    filtered_labels = []
    for doc, label in zip(corpus, labels):
        if doc.strip():
            filtered_corpus.append(doc)
            filtered_labels.append(label)

    return filtered_corpus, filtered_labels

def main():
    corpus,labels = get_data() #获取数据集
    print('总的数据量：',len(corpus))
    print('labels数据量：',len(labels))
    corpus,labels = remove_empty_docs(corpus,labels)
    print('样本之一：',corpus[0])
    print('样本的label:',labels[243])
    label_name_map = ['垃圾邮件','正常邮件']
    print('实际：',label_name_map[int(labels[10])],label_name_map[int(labels[59])])
    #对数据进行划分
    train_corpus,test_corpus,train_labels,test_labels=prepare_datasets(corpus,labels,test_data_proportion=0.3)
    print('训练数据量：',len(train_corpus))
    print('测试数据量：',len(test_corpus))

    from normalization import normalize_corpus
    #对数据归一化
    normalize_train_corpus = normalize_corpus(train_corpus)
    normalize_test_corpus = normalize_corpus(test_corpus)

    ''.strip()

    from feature_extractors import bow_extractor,tfidf_extractor
    import gensim
    import jieba

    #词袋模型特征
    bow_vectorizer,bow_train_features = bow_extractor(normalize_train_corpus)
    bow_test_features = bow_vectorizer.trainsform(normalize_test_corpus)

---------








if __name__ == '__main__':
    main()