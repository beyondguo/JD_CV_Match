#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/14 10:03
# @Author  : Frances
# @Site    : 
# @File    : entity.py
# @Software: PyCharm

from modeltools.model_interface import  cvEntity
from modeltools.model_interface import  jdEntity

# 实体抽取模块

# # cv实体抽取模块
def get_cv_entity(pre_handled_data):
    '''
    :param pre_handled_data: 预处理之后的数据
    :return:实体抽取之后的结果
    '''
    # 需要提取实体数据
    resume_process = []
    # 不需要提取
    resume_not = []
    # 返回结果
    resume = []
    for folder_data in pre_handled_data:
        resume_pro = []
        resume_no = []
        for file in folder_data: # 一个文件里的一份简历
            if '工作经验' in file.keys() or '项目经验'  in file.keys() or '自我评价' in file.keys():
                resume_pro.append(file)
            else:
                resume_no.append(file)

        resume_process.append(resume_pro)
        resume_not.append(resume_no)

    if len(resume_process): # 模型提取实体
        resumeModel = cvEntity()
        resume = resumeModel.entity_extract(resume_process)
        assert len(resume_process)==len(resume),"提取数量错误！"
    else:
        resume=resume_process+resume_not
    return resume

# # jd实体抽取模块
def get_jd_entity(pre_handled_data):
    '''

    :param pre_handled_data: 预处理之后的数据
    :return: 实体抽取之后的结果
    '''
    # 需要提取实体数据
    jd_process = []
    # 不需要提取
    jd_not = []
    # 返回结果
    jd = []
    for jds in pre_handled_data:
        if 'details' in jds.keys():
            jd_process.append(jds)
        else:
            jd_not.append(jds)

    if len(jd_process):
        jdModel = jdEntity()
        jd = jdModel.entity_extract(jd_process)
        assert len(jd_process) == len(jd), "提取数量错误！"
    else:
        jd = jd_process + jd_not
    return jd