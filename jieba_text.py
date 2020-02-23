# -*- conding: utf-8 -*-

import jieba
import jieba.posseg as psg
import datetime

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

print(str(datetime.today().strftime('%Y年%m月%d日')))