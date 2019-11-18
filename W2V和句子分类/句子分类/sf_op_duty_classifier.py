# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 11:16:55 2018

@author: x1c
"""

# =================================wrod2vec模型：============================================
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
wv = KeyedVectors.load_word2vec_format(datapath('F:/MyDownloads/merge_sgns_bigram_char300.txt/merge_sgns_bigram_char300.txt'), binary=False)
vocab = wv.vocab
import jieba
def sentence2vec(sentence):
    words = jieba.lcut(sentence)  ### 这里值得斟酌，
    print('.',end='')
    sen_v = np.zeros((300))
    n = 0
    for word in words:
        if word in vocab.keys() and word!='':
            n += 1
            word_v = wv[word]
            sen_v = sen_v+word_v
        if n>0:
            sen_v = sen_v/n
    return sen_v
# =============================================================================
     
 
# =================================Load classifier: ============================================
from keras.models import load_model
model = load_model('my_model.h5')
model.summary()
import numpy as np
def sentenceTopic(sentence):
    your_vector = np.array(sentence2vec(sentence)).reshape(1,300)
    predict = model.predict(your_vector)
    result = np.argmax(predict)
     
    if result == 0:
        print("该句的主题是:\n","【协调-统筹-汇报上级】")
    elif result == 1:
        print("该句的主题是:\n","【客户-市场】")
    elif result == 2:
        print("该句的主题是:\n","【团队-人员】")
    elif result == 3:
        print("该句的主题是:\n","【仓储-设备】")
    elif result == 4:
        print("该句的主题是:\n","【运营管理-保障-安全】")
    elif result == 5:
        print("该句的主题是:\n","【计划-分析-战略】")
# =============================================================================
sentenceTopic("团队")