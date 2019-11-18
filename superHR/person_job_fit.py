# -*- coding: utf-8 -*-
"""
Created on Tue May 14 19:13:19 2019

@author: gby
"""

# Loading W2V model:
from gensim.models import Word2Vec
import numpy as np
print("Laoding word2vec model, may take a few minutes......")
if ('wvmodel' not in vars()): # 避免重复加载  
    wvmodel = Word2Vec.load('F:/Jupyter/--NLP/big_things/models/wikibaikeWV250/wikibaikewv250')
wvdim = 250
import jieba
from sklearn.metrics.pairwise import cosine_similarity
vocab = wvmodel.wv.vocab
wvmodel['哈哈'].shape

# ================================ algorithms of 4 process type:=========================
"""Process type : 1
worlist_similarity
1. 把实体列表中的每一个实体表示成一个向量
2. 两两计算词向量相似度
3. 设定阈值，高于阈值的就拿出来计算平均分
4. 可将匹配矩阵导出用于可视化
"""
import copy
def e2v(entity): # 给任何一个任意长度的词，返回一个定长的词向量
    try:
        if entity in vocab:
            return wvmodel[entity]
        else:
            wv = np.zeros((wvdim,))
            words = jieba.lcut(entity)
            count = 0
            for word in words:
                if word in vocab:
                    count += 1
                    wv += wvmodel[word]
            if count > 0:
                return wv/count
            else:
                return np.zeros((wvdim,))
    except Exception as e:
        print('==error==:',e)
        print('==error entity==:',entity)

def wordList_similarity(entity_list1,entity_list2,threshold = 0.2): # 计算两个词列表的相似度
    wv_list1 = [e2v(entity).reshape(1,wvdim) for entity in entity_list1]
    wv_list2 = [e2v(entity).reshape(1,wvdim) for entity in entity_list2]
    E1 = np.concatenate(wv_list1) # shape:(n,wvdim), n is the length of list
    E2 = np.concatenate(wv_list2)
    similarity_matrix = cosine_similarity(E1,E2)
    highlight_matrix = copy.deepcopy(similarity_matrix)
    def show_up(x):
        x[x<threshold] = 0
        return x
    highlight_matrix = show_up(highlight_matrix)
#    print('similarity_matrix:\n',similarity_matrix)
#    print('highlight_matrix:\n',highlight_matrix)

#     highlight_items = np.squeeze(highlight_matrix.reshape(1,-1))
    highlight_items = highlight_matrix.reshape(1,-1)[0]
    sumup = 0
    count = 0
    highlight_score = 0
#    print('highlight_items:',highlight_items)
    for item in highlight_items:
        if item>0:
            sumup += item
            count += 1
    if count>0:
        highlight_score = sumup/count
#    print('Overall similarity:',np.average(similarity_matrix))
#    print('Highlight similarity:',highlight_score)
    return highlight_score

# example:
#wordList_similarity(['北京','是','中国','首都'],['巴黎','乃','法国','都城'])

"""Process type : 2
number_compare
JD和CV的一些标签按照数值大小进行比较：
1. 若CV的数字大于等于JD的要求，就匹配度为1；
2. 若CV的数字小于JD的要求，则计算gap，按照某种函数映射到0~1内，然后用1减去该值，得到匹配度；
3. 若JD的要求为空，则匹配度为1；
4. 若JD又不为空，而CV为空，则匹配度为0；
5. 此处可以设置硬卡控，则不满足要求直接匹配度为0；
"""
import math
import re
def sigmoid(x):
    return 1/(1+pow(math.e,-x))

chinese2num = {'一':1,'二':2,'两':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10,\
              '十一':11,'十二':12,'十三':13,'十四':14,'十五':15}
chineseNums = ''.join(list(chinese2num.keys()))
def get_num(sentence):
    s = str(sentence)
    num_cn = re.findall(r"(["+chineseNums+"]{1,2})",s)
#     num_ab = re.findall(r"([0-9]{1,2})",s)
    num_ab = re.findall(r'([0-9]{1,}[.][0-9]*)',s)
    if len(num_cn)>0:
        num = chinese2num[num_cn[0]]
    elif len(num_ab)>0:
        num = num_ab[0]
    else:
        num = -1
    return float(num)

def number_compare(JD_entity_list,CV_entity_list,isHardRule = False):
#    print(JD_entity_list,CV_entity_list)
    s1 = ''.join([str(x) for x in JD_entity_list])
    s2 = ''.join([str(x) for x in CV_entity_list]) # 要先都转化成str，才能够使用join连起来。原list可能含有其他类型
#     num1 = int(get_num(s1))
#     num2 = int(get_num(s2))
    num1 = get_num(s1)
    num2 = get_num(s2)
#    print('jd num:',num1,'cv num:',num2)
    if num1 == -1:
        return 1
    elif num2 == -1:
        return 0
    elif num1 <= num2:
        return 1
    elif isHardRule:
        return 0
    else:
        return 1-sigmoid(abs(num1-num2))

#example:
#jd = ['四年工作经验']
#cv = ['工作3.5年']
#number_compare_score(jd,cv,isHardRule=False)


"""Process type : 3
level_compare

"""
xueli_level = {'小学':1,'初中':2,'高中':3,'大专':4,'本科':5,'硕士':6,'博士':7}
def level_compare(JD_entity_list,CV_entity_list,tag,isHardRule = False):
    if tag=='学历':
        if len(JD_entity_list) == 0:
            num1 = -1
        else:
            num1 = xueli_level[JD_entity_list[0]]
        if len(CV_entity_list) == 0:
            num2 = -1
        else:
            num2 = xueli_level[CV_entity_list[0]]
#        print('jd xueli:',num1,'cv xueli:',num2)
        if num1 == -1:
            return 1
        elif num2 == -1:
            return 0
        elif num1 <= num2:
            return 1
        elif isHardRule:
            return 0
        else:
            return 1-sigmoid(abs(num1-num2))

    elif tag=='学校':
        return 1
    else:
        return 1  # ?????????????
    return 1

"""Process type : 4
bool_compare

"""
def bool_compare(JD_entity_list,CV_entity_list):
    set1 = set(JD_entity_list)
    set2 = set(CV_entity_list)
    inter = set1.intersection(set2)
    if len(inter)>0:
        return 1
    else:
        return 0


#  整合上面4个函数：
def entity_list_compare(JD_entity_list,CV_entity_list,process_type,tag=None,threshold=0.2):
    """
    参数:
    JD_entity_list，CV_entity_list，要比较的两个实体列表
    process_type, 可取1,2,3,4, 详见‘标准化表’
    """
    assert isinstance(JD_entity_list,list)==True, '当前输入的JD_entity_list不是list类型！'
    assert isinstance(CV_entity_list,list)==True, '当前输入的CV_entity_list不是list类型！'

    if len(JD_entity_list) == 0:
#        print('JD here is empty! Return socre 1.')
        return 1
    if len(CV_entity_list) == 0:
#        print('CV here is empty! Return score 0.')
        return 0

    if int(process_type) == 1: # 文本相似度
#        print("Using wordList_similarity method.")
        return wordList_similarity(JD_entity_list,CV_entity_list,threshold=threshold)

    elif int(process_type) == 2: # 数值大小
#        print("Using number_compare method.")
        return number_compare(JD_entity_list,CV_entity_list,isHardRule=False)

    elif int(process_type) == 3: # 层次比较
#        print("Using level_compare method.")
        return level_compare(JD_entity_list,CV_entity_list,tag)

    elif int(process_type) == 4: # 布尔比较
#        print("Using bool_compare method.")
        return bool_compare(JD_entity_list,CV_entity_list)

    else:
        print("Error: Wrong process_type! Must be 1,2,3,4! Your silly input type is:",process_type)
        return "Error: Wrong process_type! Must be 1,2,3,4!"


# ===========================字典比较函数：==========================================
"""
对比结构大致相同的JD和CV字典。
允许两个字典不一样，以JD的字典为准！
"""

## 设计一个函数，可以一口气比较两个字典。两个字典的结构完全一样,由tag-value组成
## 根据JD中的tag去找CV中对于的tag，最后各tag的分数加权平均
def dict_match(JD_dict,CV_dict,tag_weights_dict):
    tags = JD_dict.keys()
    if len(tags) == 0: # 如果JD没有tags，那就当做都满足了
        return 1
    else:
        final_score = 0 # 是各个tag得分的加权平均
        weights_sum = 0
        for tag in tag_weights_dict:
            weights_sum += tag_weights_dict[tag]
        for tag in tags:
            assert JD_dict[tag][0] == CV_dict[tag][0], '老哥，两个标签的处理类型不一样呀，请检查！'
            JD_entity_list = JD_dict[tag][1]
            CV_entity_list = CV_dict[tag][1]
            process_type = JD_dict[tag][0]
            score = entity_list_compare(JD_entity_list,CV_entity_list,process_type)
            score = score*tag_weights_dict[tag]
#            print('current tag socre:','tag',tag,score)
            final_score += score
        return final_score/weights_sum # 最后除以这个，使得范围在0~1


#================================Calculate scores of 3 dimentions:================
"""
计算“经验”维度的得分
"""
def softmax(x):
    x = np.array(x).reshape(1,len(x))
    return np.exp(x)/np.sum(np.exp(x),axis=1)
def desc_weights(length): # 递减权重，用于多段经历
    w = sorted([i for i in range(1,length+1)],reverse=True)
    return list(softmax(w)[0])

## tag的权重事先给定一个初始值
#exp_tag_weights = {'公司名':1, '所在城市':1, '业务':1, '产品':1, '职位':1, '年限':1, '行业':1}
"""
计算经验得分，要在内部区分不同时间段的经验
"""
def get_exp_score(jd_exp,cv_exp,tag_weights):
    """
    直接从标准化输入中，根据‘经验’的key来取出。
    jd_exp应该直接是一个字典，包含标签和对应的实体；
    cv_exp
    jd_exp,cv_exp都是字典，前者单层，后者多层
    """
    jd_exp = jd_exp
#    tags = list(jd_exp.keys())

    allType_scores  =[]
    for exp_type in ['工作经验','项目经验']: # 各自都是一个list，包含若干个dict
        # 对每一种经验，计算其各段子经验的加权平均
#        print("------------Now processing %s----------"%exp_type)
        exps = cv_exp[exp_type] # 一个list，包含若干个dict
        if len(exps)>0:
            time_weights = desc_weights(len(exps))
            current_type_score = 0
            for t,exp in enumerate(exps):
#                print('exp no.',t)
#                print('score:',dict_match(jd_exp,exp,tag_weights))
#                print('time weight:',time_weights[t])
                current_type_score += dict_match(jd_exp,exp,tag_weights)*time_weights[t]
            allType_scores.append(current_type_score)
        else:
            allType_scores.append(0)
#    print("------------Now processing %s----------"%'其他经验')
    allType_scores.append(dict_match(jd_exp,cv_exp['其他经验'],tag_weights)) # 其他经验不是list，就是一个dict，所以直接丢进入算
#    print('score:',dict_match(jd_exp,cv_exp['其他经验'],tag_weights))
    best_score = max(allType_scores)
    return best_score

#get_exp_score(jd_words['经验'],cv_words['经验'],tag_weights=exp_tag_weights)

"""
计算“技能”维度的得分
这个最简单，就是一个list
"""
def get_skill_score(jd_skill,cv_skill):
    process_type = int(jd_skill[0])
    jd_entity_list = jd_skill[1]
    cv_entity_list = cv_skill[1]
    score = entity_list_compare(jd_entity_list,cv_entity_list,process_type)
    return score
#skill_score(jd_words['技能'],cv_words['技能'])

"""
计算教育维度得分，当前只考虑最近一段教育经历
"""
#edu_tag_weights = {'专业':1,'学历':1,'学校':1}
def get_edu_score(jd_edu,cv_edu,tag_weights):
    return dict_match(jd_edu,cv_edu[0],tag_weights)

#edu_score(jd_words['教育'],cv_words['教育'],edu_tag_weights)


# ====================================Overall Score:==========================
def overall_score(jd,cv,exp_tag_weights,edu_tag_weights,dim_weights={'经验':1,'技能':1,'教育':1},print_score=False):
    exp_score = get_exp_score(jd['经验'],cv['经验'],exp_tag_weights)
    skill_score = get_skill_score(jd['技能'],cv['技能'])
    edu_score = get_edu_score(jd['教育'],cv['教育'],edu_tag_weights)
    weights_sum = dim_weights['经验']+dim_weights['技能']+dim_weights['教育']
    overallscore = (exp_score*dim_weights['经验']+skill_score*dim_weights['技能']+edu_score*dim_weights['教育'])/weights_sum
    if print_score:
        print('======================Result:========================')
        print('JD name: ',jd['name'],' JD id: ',jd['id'])
        print(' CV id: ',cv['id'])
        print('经验维度得分：',exp_score)
        print('技能维度得分：',skill_score)
        print('教育维度得分：',edu_score)
        print('总分：',overallscore)
    return overallscore





