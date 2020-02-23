# -*- coding: UTF-8 -*-
#实现基于规则的分词
#逆向最大匹配法
import util.dic

class RMM():
    def __init__(self):
        self.windows_size = 4
    def cut(self,text):
        result =[]
        dic = util.dic.get_dic(self)
        txt_len = len(text)
        index = len(text)
        #
        while  index>0:
            for size in range(index-self.windows_size, index):
                # print(size)
                piece = text[size:index]
                # print(piece + "---")
                if (piece in dic):
                    # print(piece)
                    index = size + 1
                    break
            index = index - 1
            result.append(piece)
        result.reverse()
        return result
