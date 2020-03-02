该文件夹下是命名实体识别 测试项目



使用crf++训练模型
--训练
.\crf_learn.exe  -f 4 -p 8 -c 3  template  E:\project\test_nlp\NER\data\train.txt  model
--测试
 .\crf_test.exe -m .\model E:\project\test_nlp\NER\data\test.txt > E:\project\test_nlp\NER\data\test.rst