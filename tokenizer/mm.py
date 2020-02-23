# -*- coding: UTF-8 -*-
#实现基于规则的分词
#正向最大匹配法
import util.dic

class MM():
    def __init__(self):
        self.windows_size=4

    #正向最大匹配法
    def cut(self,text):
        result = []
        #加载词典
        dic=util.dic.get_dic(self)
        # print(dic)
        txt_len = len(text)
        index = 0
        #
        while txt_len>index:
          for size in range(self.windows_size+index,index,-1):
              # print(size)
              piece = text[index:size]
              # print(piece+"---")
              if(piece in dic):
                  # print(piece)
                  index = size-1
                  break
          index = index+1
          result.append(piece)
        return result



