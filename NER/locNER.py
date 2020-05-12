# -*- coding: UTF-8 -*-
#加载CRF++训练的模型使用python调用
#命名实体识别
def load_model(path):
    import os,CRFPP
    if os.path.exists(path):
        return CRFPP.Tagger('-m {0} -v 3 -n2'.format(path))
    return None
def locationNER(text):
    tagger = load_model('model')
    for c in text:
        tagger.add(c)
        result = []
        tagger.parse()
        word = ''
        for i in range(0,tagger.size()):
            for j in  range(0,tagger.xsize()):
                ch = tagger.x(i,j)
                tag = tagger.y2(i)
                if tag =='B':
                    word = ch
                elif tag == 'M':
                    word +=ch
                elif tag == 'E':
                    word += ch
                    result.append(word)
                elif tag == 'S':
                    word = ch
                    result.append(word)
    return result



if __name__ == '__main__':
    text = '我中午要去北京饭店，下午去中山公园，晚上回亚运村。'
    print(text, locationNER(text), sep='==> ')

    text = '我去回龙观，不去南锣鼓巷'
    print(text, locationNER(text), sep='==> ')

    text = '打的去北京南站地铁站'
    print(text, locationNER(text), sep='==> ')

    text = '郑州这座城市马上就要换发新机'
    print(text, locationNER(text), sep='==> ')