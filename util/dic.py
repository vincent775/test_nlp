# -*- coding: UTF-8 -*-
#获取字典
class dic():
    dic =[]
    def __init__(self):
        self.dic = self.get_dic()
    def get_dic(self):
        dict=[]
        # read txt method one
        f = open("./data/dic.txt",encoding='UTF-8')
        line = f.readline()
        while line:
            # print(line)
            dict.append(line.rstrip())
            line = f.readline()
        f.close()
        return dict