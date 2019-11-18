# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 17:08:00 2018

@author: x1c
"""
import fastText.FastText as ff


import pandas as pd
raw_data = pd.read_excel('hr-产品经理-句子样本(1).xls',sheetname='Sheet3')
sentences = raw_data['sentence']
labels = raw_data['class']

with open('hr_pm_train.txt','w',encoding='utf-8') as f:
    for sentence,label in zip(sentences[0:650],labels[0:650]):
        f.writelines(sentence+"    	__label__"+str(label)+"\n")

with open('hr_pm_test.txt','w',encoding='utf-8') as f:
    for sentence,label in zip(sentences[651:766],labels[651:766]):
        f.writelines(sentence+"    	__label__"+str(label)+"\n")

model = ff.train_supervised('hr_pm_train.txt')

result = model.test('hr_pm_test.txt')
print("precision:",result[1])
print("recall:",result[2])


print(model.predict('准备:调研可行性和需求进行调研后,有可实施改进空间'))

print(model.get_labels())
