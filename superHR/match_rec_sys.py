# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:46:34 2019

@author: gby
"""

from person_job_fit import overall_score
import pandas as pd
import json
import os


ground_truth_path = 'data/roche100_id_score.xlsx'
# SrAccountExecutive , ApplicationsSpecialist
jd_title = 'SrAccountExecutive'
jd_path = 'data/jd/%s.json'%jd_title
cv_path = 'data/cv/output_norm_200/'+jd_title



def match(jd_path,cv_path,exp_tag_weights,edu_tag_weights,dim_weights):
    #读取jd：
    with open(jd_path,encoding='utf-8') as f:
        jd_words = json.loads(f.read())
#        print('JDJDJDJDJDJD_*_*_*_*_*_*_*_*_*_*')
#        for each in jd_words:
#            print(each)

    # 读取所有cv：
    cv_names = os.listdir(cv_path)
    cv_ids = []
    cv_scores = []
    for cv_name in cv_names:
        with open(cv_path+'/'+cv_name,encoding='utf-8') as f:
            cv_words = json.loads(f.read())

        cv_id = cv_words['id']
        score = overall_score(jd_words,cv_words,exp_tag_weights,edu_tag_weights,dim_weights)
        cv_ids.append(cv_id)
        cv_scores.append(score)

    result = pd.concat([pd.DataFrame(cv_ids),pd.DataFrame(cv_scores)],axis=1)
    result.columns = ['id','score']
    result = result.sort_values(by='score',ascending=False)
#    result.to_csv('data/results/%s_predict.csv'%jd_title)

    return result

#=========================加载ground truth数据，评价效果=========
truth = pd.read_excel(ground_truth_path,sheet_name=jd_title)


def topn_index(rank1,rank2,n,candidate_size):
    set1 = set(rank1[:n])
    set2 = set(rank2[:n])
    inter = set1.intersection(set2)
    gain = (len(inter)/n - n/candidate_size)/(n/candidate_size)
    return [len(inter)/n,gain]


# 通过grid-search来寻找最优的参数组合：
with open('data/results/grid.txt','w') as f:
    param_range = 6
    for a in range(1,param_range+1):
        for b in range(1,param_range-a+1):
            for c in range(1,param_range-a-b+1):

                print('\n===Params:a:%d,b:%d,c:%d'%(a,b,c),'\n')
                f.write('\n===Params:业务:%d,职位:%d,行业:%d\n'%(a,b,c))

                exp_tag_weights = {'公司名':0, '所在城市':0, '业务':a, '产品':0, '职位':b, '年限':0, '行业':c}
                edu_tag_weights = {'专业':1,'学历':1,'学校':1}
                dim_weights = {'经验':3,'技能':2,'教育':1}
                result = match(jd_path,cv_path,exp_tag_weights,edu_tag_weights,dim_weights)
                overall_gain = 0
                for n in [5,10,20,30,50]:
                    rank1 = list(truth.id)
                    rank2 = list(result.id)
                    topn_score,topn_gain = topn_index(rank1,rank2,n,100) #??candidate_size??
                    overall_gain += topn_gain
                    plus = ''
                    if topn_gain>0:
                        plus = '  ■'
                    print('top%d-score:%.2f, gain:%.2f%s'%(n,round(topn_score,2),round(topn_gain,2),plus))
                    f.write('top%d-score:%.2f, gain:%.2f%s\n'%(n,round(topn_score,2),round(topn_gain,2),plus))
                post = ''
                if overall_gain>0:
                    post = '   ●●●●●这个不错●●●●'
                print('overall_gain:%.2f%s'%(overall_gain,post))
                f.write('--Overall_gain: %.2f%s\n'%(overall_gain,post))


