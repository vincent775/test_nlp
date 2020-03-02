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

if __name__ == '__main__':
    f1('./data/test.rst')




