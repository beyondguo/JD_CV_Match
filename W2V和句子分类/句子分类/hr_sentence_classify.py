# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 11:55:18 2018

@author: GBY
"""

## Load raw data from excel:
import pandas as pd
raw_data = pd.read_excel('hr-产品经理-句子样本(1).xls',sheetname='Sheet3')

## Load my word2vec model:
from gensim.models import Word2Vec
wv_model = Word2Vec.load('../models/wikibaikeWV250/wikibaikewv250')
vocab = wv_model.wv.vocab

## Define a sentence-to-vector function:
import jieba
jieba.load_userdict('../new_dict.txt')
import numpy as np
def sentence2vec(sentence):
    words = jieba.lcut(sentence)
    s_v = np.zeros(250)  ## My word3vec model is of 250 dementions.
    n = 0
    for word in words:
        if word in vocab.keys():
            w_v = wv_model[word]
            s_v += w_v
            n += 1
    if n>0:
        s_v /= n
    return s_v


import keras
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split,KFold
## Split X,Y from raw data,and split out the training and test set:
## Convert the sentences to vectors and save them to dataframe:
X_sentences = raw_data['sentence']
X_vectors = []
for each in X_sentences:
    s_v = sentence2vec(str(each))
    X_vectors.append(s_v)
Y_class = raw_data['class']

X = np.array(X_vectors)
Y = keras.utils.to_categorical(Y_class,2)


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33,random_state=4)

kf = KFold(n_splits=10)


## Build a NN model:
model = Sequential()
model.add(Dense(32,activation='relu',input_shape=(250,)))
model.add(Dense(16,activation='relu'))
model.add(Dense(2,activation='sigmoid'))

model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train,Y_train,epochs=2000,batch_size=64)

model.evaluate(X_test,Y_test)


## 10-fold cross validation:
# =============================================================================
# accuracy = []
# for train_index,test_index in kf.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     Y_train, Y_test = Y[train_index], Y[test_index]
#
#     hitory = model.fit(X_train,Y_train,epochs=1,batch_size=62)
#     test_accuracy = model.evaluate(X_test,Y_test)[1]
#     accuracy.append(test_accuracy)
#     print('------------',test_accuracy)
#
# print(accuracy)
# print(np.mean(np.array(accuracy)))
#
#
# X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33)
# model.evaluate(X,Y)
# =============================================================================


model.save('hr-pm.h5')

def predictClass(sentence):
    my_example = np.array(sentence2vec(sentence)).reshape(1,250)
    return np.argmax(model.predict(my_example))

