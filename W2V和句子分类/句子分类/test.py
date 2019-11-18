# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 18:32:48 2018

@author: x1c
"""

from beyondnlp import *
import time
import numpy as np
from pandas.core.frame import DataFrame
import pandas as pd
################句子转化成向量##############
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
wv = KeyedVectors.load_word2vec_format(datapath('F:/MyDownloads/merge_sgns_bigram_char300.txt/merge_sgns_bigram_char300.txt'), binary=False)
vocab = wv.vocab
def sentence2vec(sentence):
    words = text2words(sentence)
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


##### Get the training set and test set! #############
v_c_pd = pd.read_csv('new_op_v_c.csv')
#------用sklearn的包来划分：
from sklearn.model_selection import train_test_split
X = np.array(v_c_pd.loc[:,'0':'299'])
Y = np.array(v_c_pd.loc[:,'class'])
X_train, X_test, Y_train, Y_test = train_test_split(
                           X,Y,test_size=0.33, random_state=42)

#########################下面的划分方法太蠢了：
train1 = v_c_pd[0:209]
test1 = v_c_pd[209:310]
train2 = v_c_pd[310:360]
test2 = v_c_pd[360:379]
train4 = v_c_pd[379:439]
test4 = v_c_pd[439:466]
train5 = v_c_pd[466:642]
test5 = v_c_pd[642:712]
train6 = v_c_pd[712:832]
test6 = v_c_pd[832:892]
train7 = v_c_pd[892:1192]
test7 = v_c_pd[1192:1360]

train_set = pd.concat([train1,train2,train4,train5,train6,train7],axis=0)
test_set = pd.concat([test1,test2,test4,test5,test6,test7],axis=0)

train_set.to_csv('op_train',sep=',')
test_set.to_csv('op_test',sep=',')

x_train = train_set.loc[:,'0':'299'] # loc[行，列]
y_train = train_set.loc[:,'class']
x_test = test_set.loc[:,'0':'299']
y_test = test_set.loc[:,'class']

x_train = np.array(x_train).reshape(915,300)
y_train = np.array(y_train).reshape(915,1)
x_test = np.array(x_test).reshape(445,300)
y_test = np.array(y_test).reshape(445,1)
###########################################

###  Using a simple NN to train the model:  ######
import keras
from keras.models import Sequential,Model
from keras.layers import Dense,Activation,Dropout

num_class = 6
Y_train_onehot = keras.utils.to_categorical(Y_train,num_class)
Y_test_onehot = keras.utils.to_categorical(Y_test,num_class)

### -----------------------define your model:
model = Sequential()
model.add(Dense(32,activation='relu', input_shape=(300,)))
model.add(Dense(num_class,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#### -----------------------Start training!!
t1 = time.time()
history = model.fit(X_train,Y_train_onehot,epochs=100,batch_size=64)
t2 = time.time()
print("Training time:",t2-t1,"s")
losses = history.history['loss']
import matplotlib.pyplot as plt

plt.figure()
plt.plot(losses)
plt.title("2-layer NN on Operation data")
plt.ylabel("Loss")
plt.xlabel("epoches")
plt.show()

#### --------------------------Test it!  

########## Confusion matrix：
from sklearn.metrics import confusion_matrix
y_pred = np.argmax(model.predict(x_test),axis=1)
y_test_orig = np.array(test_set.loc[:,'class'])
c_m = confusion_matrix(y_pred,y_test_orig)

score = model.evaluate(X_test,Y_test_onehot)
print("Total loss:",score[0])
print("Test accuracy:",score[1])
import h5py


#########################################################

###### ------------- Use your own example to test it :
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

#your_vector = np.array(sentence2vec("负责协调跨部门")).reshape(1,300)
#predict = model.predict(your_vector)
#result = np.argmax(predict)
my_sentences = ['协同各部门开展整合的具体实施工作','协助组织召开区部职能例会、企划例会及月度、季度、年度经营分析会',\
                '及时了解客户期望','执行点部大客户维护及协助开发','包括市场趋势研究',\
                '执行对点部人员的业务培训','负责中转场人员管理及团队建设','负责组织并指导运维人员保证系统的正常运营、信息的综合利用及系统的安全性',\
                '"1、主导华南RDC仓储精益改善、流程优化、成本降低等重点项目落地','承担分、点部的安全管理工作',\
                '负责对技术研发团队进行有效管理、监督和考核','协助开展市场调查并配合做好公司产品的宣传工作','为业务发展提供决策支持"']
for each in my_sentences:
    print(each)
    sentenceTopic(each)
    print('-------------')


################### 保存模型：-----------------
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
from keras.models import load_model
model = load_model('my_model.h5')
model.summary()
sentenceTopic("协助开展市场调查并配合做好公司产品的宣传工作")
######################################### Run these:###########
segmentor.release()
postagger.release()
parser.release()