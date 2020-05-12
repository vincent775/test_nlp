# -*- condig: UTF-8 -*-
#手动编写提取关键词
#主要的方式有：TF-IDF、TextRank、主题模型（LSI、LDA）

import math
import jieba
import jieba.posseg as psg
from  gensim import corpora,models
from jieba import analyse
import functools

# 加载停用词
def get_stopword_list():
    #停用词表存储路径为每行一个词，按航读取进行加载
    #进行编码转换确保匹配率
    stop_word_path = './stopword.txt'
    stopword_list = [sw.replace('\n','') for sw in open(stop_word_path).readline()]
    return stopword_list
#去除干扰词
def word_filter(seg_list,pos=False):
    stopword_list =  get_stopword_list()
    filter_list = []
    #根据POS参数选择是否词性过滤
    ##不进行词性过滤，则将词性都标记为n，表示全部保留
    for seg in seg_list:
        if not pos:
            word = seg
            flag = 'n'
        else:
            word =  seg.word
            flag = seg.flag
        if not word in stopword_list and len(word)>1:
            filter_list.append(word)
    return filter_list
