# -*- condig: UTF-8 -*-
# 使用crf++训练模型
# --训练
# .\crf_learn.exe  -f 4 -p 8 -c 3  template  E:\project\test_nlp\NER\data\train.txt  model
# --测试
#  .\crf_test.exe -m .\model E:\project\test_nlp\NER\data\test.txt > E:\project\test_nlp\NER\data\test.rst
#本次的代码是使用crf++ 训练好的模型，并且进行测试之后生成的结果 .rst 文件，进行测试效果展示

def f1(path):
    with open(path,'r+',encoding='utf-8') as f:
        all_tag = 0 #记录所有的标记数
        loc_tag =  0 # 记录真实地理位置标记数
        pred_loca_tag = 0 #记录预测的地理位置标记数
        correct_tag = 0 # 记录正确的标记数
        correct_loc_tag = 0 #记录正确的地址位置标记数
        # print(f.readline(1))
        states =['B','M','E','S']
        for line in f:
            line = line.strip()
            if line =='': continue
            _,r,p = line.split()
            all_tag +=1
            if r==p:
                correct_tag +=1
                if r in states :
                    correct_loc_tag +=1
            if r in  states: loc_tag+=1
            if p in states: pred_loca_tag +=1
        loc_p = 1.0*correct_loc_tag/pred_loca_tag
        loc_R = 1.0*correct_loc_tag/loc_tag
        print('loc_P:{0},loc_R:{1},loc_F1:{2}'.format(loc_p,loc_R,(2*loc_p*loc_R)/(loc_p+loc_R)))

def load_model(path):
    import os,CRFPP
    if os.path.exists(path):
        return CRFPP.Tagger('-m {0} -v 3 -n2'.format(path))
    return None

def locationNER(text):
    tagger = load_model('./model')

    for c in text:
        tagger.add(c)
    result = []

    tagger.parse()
    word = ''
    for i  in range(0,tagger.size()):
        for j in range(0,tagger.xsize()):
            ch = tagger.x(i,j)
            tag = tagger.y2(i)
            if tag =='B':
                word = ch
            elif tag == 'M':
                word += ch
            elif tag == 'E':
                word += ch
                result.append(word)
            elif tag =='S':
                word = ch
                result.append(word)
    return result


if __name__ == '__main__':
    f1('./data/test.rst')

    text ='我中午要去北京饭店'
    text = '我中午要去北京饭店，下午去中山公园，晚上回亚运村。'
    print(text, locationNER(text), sep='==> ')

    text = '我去回龙观，不去南锣鼓巷'
    print(text, locationNER(text), sep='==> ')

    text = '打的去北京南站地铁站'
    print(text, locationNER(text), sep='==> ')




