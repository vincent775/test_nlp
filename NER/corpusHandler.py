# -*- coding: UTF-8 -*-
#比如我们现在进行地名的识别，那么地名在词性标注中就是表示为ns。所以我们把词性标注的语料进行提取，提取为CRF能够识别的模式。
#例如： 中国/ns —/w 东盟/ns 首脑/n 非正式/b 会晤/vn。
# 我们就是要提取 ns 词性的，分为训练集和测试集。
def tag_line(words,mark):
    chars = []
    tags = []
    temp_word ='' #用于合并组合词
    for word in words:
        word = word.strip('\t')
        if temp_word == '':
            bracket_pos = word.find('[')
            w,h = word.split('/')
            if bracket_pos == -1:
                if len(w) == 0: continue
                chars.extend(w)
                if h == 'ns':
                    tags += ['S'] if len(w)==1 else ['B']+['M']*(len(w)-2)+['E']
                else:
                    tags +=['O']*len(w)
            else:
                w = w[bracket_pos+1:]
                temp_word = w
        else:
            bracket_pos =  word.find(']')
            w,h = word.split('/')
            if bracket_pos == -1:
                temp_word   += w
            else:
                w = temp_word+w
                h = word[bracket_pos+1:]
                temp_word = ''
                if len(w) == 0:continue
                chars.extend(w)
                if h == 'ns':
                    tags +=['S'] if len(w) ==1 else ['B']+ ['M']*(len(w)-2)+['E']
                else:
                    tags += ['O']*len(w)
    assert temp_word ==''
    return (chars,tags)

def corpusHandler(corpusPath):
    import os
    root = os.path.dirname(corpusPath)
    with open(corpusPath,encoding='utf-8') as corpus_f,\
        open(os.path.join(root,'train.txt'),'w') as train_f, \
        open(os.path.join(root, 'test.txt'), 'w') as test_f:

        pos =0
        for line in corpus_f:
            line = line.strip('\r\n\t')
            if line == '':continue
            isTest = True if  pos%5==0 else False #抽样20%进行测试集使用
            words = line.split()[1:]
            if len(words) == 0: continue
            line_chars,line_tags = tag_line(words,pos)
            saveObj = test_f if isTest else train_f
            for k,v in enumerate(line_chars):
                saveObj.write(v + '\t' + line_tags[k] + '\n')
            saveObj.write('\n')
            pos +=1

if __name__ =='__main__':
    corpusHandler('./data/people-daily.txt')



