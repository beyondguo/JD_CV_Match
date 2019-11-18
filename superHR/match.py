"""
匹配度计算模块
"""
print("importing match module...............")
from gensim.models import Word2Vec
import numpy as np
wvmodel = Word2Vec.load('F:/Jupyter/--NLP/big_things/models/wikibaikeWV250/wikibaikewv250')
wvdim = 250
import jieba
from sklearn.metrics.pairwise import cosine_similarity
vocab = wvmodel.wv.vocab

# 输入一组数，返回对应的归一化后的结果
def softmax(x): 
    x = np.array(x).reshape(1,len(x))
    return np.exp(x)/np.sum(np.exp(x),axis=1)

# 给定列表长度，返回递减的权重列表
def desc_weights(length): 
    w = sorted([i for i in range(1,length+1)],reverse=True)
    return list(softmax(w)[0])

def Wordlist_Wv(wordlist,wvdim=wvdim,weights_descend=False):
    
    if weights_descend:
        weights = desc_weights(len(wordlist))
    else:
        weights = [1 for _ in wordlist]
    l = 0
    wv = np.zeros((wvdim,))
    for word,weight in zip(wordlist,weights):
        if word in vocab.keys():
            wv += wvmodel[word]*weight
            l += 1
        else:
            split_words = jieba.lcut(word)
            split_weight = weight/len(split_words)
            for each in split_words:
                try:
                    wv += wvmodel[each]*split_weight
                    l += 1
                except:
                    pass
#                     print('* Warning：Word [',each,'] not in vocab!')
    if l>0: # 防止有的tag里面是空的
        return (wv/l).reshape(1,wvdim)
    else:
        return wv.reshape(1,wvdim)

"""
方法一(Baseline)：Simple AvgWV Similarity
对所有词语，进行词向量平均，然后计算cos相似度。
注：词典中没有的词，经过jieba分词后，再录入。
"""

def AvgWvSim(dic1,dic2):
    words1 = []
    words2 = []
    for li in dic1.values():
        words1 += li
    for li in dic2.values():
        words2 += li
        
    wv1 = Wordlist_Wv(words1)
    wv2 = Wordlist_Wv(words2)
    if (wv1 == np.zeros((1,wvdim))).all() or (wv2 == np.zeros((1,wvdim))).all():
        return 0
    else:
        return cosine_similarity(wv1,wv2)[0][0]

# print("JD1与CV1的匹配得分：",AvgWvSim(JD1_dic,CV1_dic))

"""
方法二：Focused-AvgW2V
对应的标签进行相似度计算，然后再按照对不同的标签的权重进行加权平均，得到总分。
（相当于添加了sentence-level attention）
"""
tag_weights = {'技能':0.3,'学历':0.1,'专业':0.2,'行业':0.2,'职能':0.1,'年限':0.1}
def FocusedAvgWvSim(dic1,dic2,tag_weights=tag_weights,weights1_desc=False,weights2_desc=False):
    total_score = 0
    for cate in dic1:
        wv1 = Wordlist_Wv(dic1[cate],weights_descend=weights1_desc)
        wv2 = Wordlist_Wv(dic2[cate],weights_descend=weights2_desc)
        if (wv1 == np.zeros((1,wvdim))).all() or (wv2 == np.zeros((1,wvdim))).all():
            score = 0
        else:
            score = cosine_similarity(wv1,wv2)[0][0]
        weighted_score = score*tag_weights[cate]
        total_score += weighted_score
    return total_score

# print("JD1与CV1的匹配得分：",FocusedAvgWvSim(JD1_dic,CV1_dic)[0][0])

