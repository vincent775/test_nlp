# -*- coding: UTF-8 -*-
import  tokenizer


if __name__ == "__main__":
    #正向最大匹配
    # tokenizer2 = tokenizer.MM()
    # print(tokenizer2.cut('有意见分歧'))
    # #
    # # # #逆向最大匹配
    # tokenizer3 = tokenizer.RMM()
    # print(tokenizer3.cut('有意见分歧'))
    #
    # #双向最大匹配
    # tokenizer4 = tokenizer.DMM()
    # print(tokenizer4.cut('研究生命的起源'))
    #
    # print(tokenizer4.cut('南京长江大桥'))
    # print(tokenizer4.cut('结婚的和尚未结婚的'))
    # print(tokenizer4.cut('有意见分歧'))

    #HMM分词
    tokenizer5 = tokenizer.HMM()
    tokenizer5.train('./data/trainCorpus.txt_utf8')
    res=tokenizer5.cut('吹筒子的吹一个外面包了火炽练蛇皮的竹筒')
    print(str(list(res)))


