# -*- conding: utf-8 -*-

import jieba
import jieba.posseg as psg
import datetime
from jieba import analyse

#结巴词性标注
sent = "六月12"
seg_list = psg.cut(sent)
print(' '.join(['{0}/{1}'.format(w,t) for  w,t in seg_list]))

#分词-全模式
seg_list = jieba.cut(sent, cut_all= True)
print('/'.join(seg_list))
#分词-精确模式
seg_list = jieba.cut(sent, cut_all= False)
print('/'.join(seg_list))
#分词-搜索引擎模式
seg_list = jieba.cut_for_search(sent)
print('/'.join(seg_list))

# print(str(datetime.today().strftime('%Y年%m月%d日')))


#结巴词性标注
#加在自定义词典
jieba.load_userdict("dict.txt")
sent = '中文分词是文本处理中不可或缺的一部分，魏亚通今天很不错，支静阳今天也很不错'

seg_list = psg.cut(sent)

print(' '.join(['{0}/{1}'.format(w,t) for w,t in seg_list]))


#结巴关键词提取技术
# 引入TF-IDF关键词抽取接口
tfidf = analyse.extract_tags

# 原始文本
text = "线程是程序执行时的最小单位，它是进程的一个执行流，\
        是CPU调度和分派的基本单位，一个进程可以由很多个线程组成，\
        线程间共享进程的所有资源，每个线程有自己的堆栈和局部变量。\
        线程由CPU独立调度执行，在多CPU环境下就允许多个线程同时运行。\
        同样多线程也可以实现并发操作，每个请求分配一个线程来处理。"

# 基于TF-IDF算法进行关键词抽取
keywords = tfidf(text)
print("keywords by tfidf:")
# 输出抽取出的关键词
for keyword in keywords:
    print(keyword + "/")


textrank = analyse.textrank
print("\nkeywords by textrank:")
# 基于TextRank算法进行关键词抽取
keywords = textrank(text)
# 输出抽取出的关键词
for keyword in keywords:
    print(keyword + "/")