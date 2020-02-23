# -*- coding: UTF-8 -*-
#实现基于规则的分词
#双向最大匹配法
#一、分词个数不同
#1.根据分词个数，一般词数较少的为优
#二、分词个数相同
#1.分词结果相同就没有什么异议，返回任意一个就行
#2.分词结果不同，返回其中单子较少的那个（比如‘研究生命的意义’）

import tokenizer


class DMM():
    def cut(self,text):
        result=[]
        mm  = tokenizer.MM()
        rmm = tokenizer.RMM()
        mm = mm.cut(text)
        rmm = rmm.cut(text)
        mm_len=len(mm)
        rmm_len=len(rmm)
        #差集
        print(list(set(mm).difference(set(rmm))))

        if mm_len<rmm_len:
            result=mm
        elif rmm_len<mm_len:
            result =rmm
        else: #两个分词次数结果相同
            mi=0
            ri=0
            for x in mm:
                if len(x)==1: mi=mi+1
            for y in rmm:
                if len(y)==1: ri=ri+1
            if mi<ri: #取单字个数较少的分词结果
                result=mm
            else: #如果单字个数相同就任意返回一个
                result = rmm
        return result



