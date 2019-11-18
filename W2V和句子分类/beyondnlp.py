
# coding: utf-8

# ## BeyondNLP：
# 
# ## 注：
# **需要把模型放在指定位置，或者修改模型位置。**
# 
# #### 主要功能：
# 1. txt文件/已经读入的文本----->句子、词（经过去空、去重处理）：
# 
# - def file2sentences(filepath)
# - def text2sentences(text)
# - def file2words(filepath,stopwords_list=False)
# - def text2words(text,stopwords_list=False)
# 
# 上面的各函数，均返回相应的list.
# 
# 2. 各种基本的NLP功能：
# 
# - 给词list进行词性标注：
#   def words2tags(words):
#   返回tags的list.
#   
# - 给文本进行句法分析，提取想要的词组结构：
#   def getRelationPhrase(text,relation_list=False,stopwords_list=False)
#   返回一个list.


import re
from pyltp import SentenceSplitter,Segmentor,Postagger,Parser

import os
LTP_DATA_DIR = 'F:/MyDownloads/ltp_data_v3.4.0/ltp_data_v3.4.0'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`



## 清理文本中的各种标点：？？？？？？？？？？？？？？？？？？？？？？？？？好像没用
def get_clean_text(text):
    text = text.strip()
    text = re.sub(r'^(1?[0-9]\.?、?\)?）?)?\*?-?·?•?(\\t)?', '', text)
    text = re.sub(r'[;；。]$', '', text)
    text = re.sub(r'^\s*', '', text)
    text = text.replace(' ','')
    return text

## 从停用词表文件中，获取停用词列表：
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

###################### 分 句：----------------------------------

## 文件-->句子
def file2sentences(filepath):
    with open(filepath,'r',encoding='utf-8') as f:
        text = f.read()
        sentences = SentenceSplitter.split(text)
        sentences = [x for x in sentences if x != ''] #去空值
    return list(set(sentences)) #去重

## text-->句子
def text2sentences(text):
    sentences = SentenceSplitter.split(text)
    sentences = [x for x in sentences if x != ''] #去空值
    return list(set(sentences)) #去重

################# 分 词：----------------------------------------
# 对于分词，可以传入一个停用词列表，不写的话默认没有。

## 文件-->词：
segmentor = Segmentor() #初始化实例
segmentor.load(cws_model_path) #加载模型
postagger = Postagger()
postagger.load(pos_model_path)
parser = Parser()
parser.load(par_model_path)
#segmentor.release() #释放模型             就暂时不释放资源了
#postagger.release()
#parser.release()
def file2words(filepath,stopwords_list=False):
    
    with open(filepath,'r',encoding='utf-8') as f:
        text = f.read()
        words = segmentor.segment(text)
        if stopwords_list == False:
            words = [x for x in words if x != ''] #去空值
        else:
            words = [x for x in words if x != '' and x not in stopwords_list]
    return list(set(words)) #去重

## text-->词
def text2words(text,stopwords_list=False):
#    segmentor = Segmentor() #初始化实例
#    segmentor.load(cws_model_path) #加载模型
    words = segmentor.segment(text)
    if stopwords_list == False:
        words = [x for x in words if x != ''] #去空值
    else:
        words = [x for x in words if x != '' and x not in stopwords_list]
#    segmentor.release() #释放模型
    return list(set(words)) #去重

############## 词 性：----------------------------------
## 根据词list获取对应的词性：
def words2tags(words):
    tags = list(postagger.postag(words))
    return tags

############## 句 法 分 析 （获取指定的词语结构，如动宾短语）：----------------------------------
def getRelationPhrase(text,relation_list=False,stopwords_list=False):

    words = text2words(text=text,stopwords_list=stopwords_list)
    tags = words2tags(words)
    result = parser.parse(words,tags)
    relation_phrase = []
    for i,each in enumerate(result):
        one = words[each.head-1]
        two = words[i]
        phrase = one+"--"+two
        if relation_list == False:
            relation_phrase.append(phrase)
        else:
            if each.relation in relation_list:
                relation_phrase.append(phrase)
    
    return relation_phrase

print("You have imported 'beyondnlp' !")
