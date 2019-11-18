#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/14 10:16
# @Author  : Frances
# @Site    : 
# @File    : main_new.py
# @Software: PyCharm

from data_pre_handle import cv_data_pre_handle,jd_data_pre_handle
import entity
import data_post_handle

# CV数据抽取
def get_cv_data(in_file_path,out_file_path,out_type):
    '''
    CV命名实体提取
    :param in_file_path: 输入数据路径
    :param out_file_path: (若输出)输出文件路径
    :param out_type: 输出类型
    :return:
    '''
    ## TODO:运行时间测试
    # 前处理模块
    pre_handled_data = cv_data_pre_handle(in_file_path)
    # 中间模块:调用模型，实体抽取
    entity_result =entity.get_cv_entity(pre_handled_data)
    print(entity_result)
    # 后处理模块
    output = data_post_handle.data_output_handle(entity_result, out_file_path, out_type)
    pass

# JD数据抽取
def get_jd_data(in_file_path,out_file_path,out_type):
    '''
    JD命名实体提取
    :param in_file_path: 输入数据路径
    :param out_file_path: (若输出)输出文件路径
    :param out_type: 输出类型
    :return:
    '''
    # 前处理模块
    pre_handled_data = jd_data_pre_handle(in_file_path)
    # 中间模块:调用模型，实体抽取
    entity_result = entity.get_jd_entity(pre_handled_data)
    # 后处理模块
    output = data_post_handle.data_output_handle(entity_result, out_file_path, out_type)
    pass

if __name__=="__main__":
    ####################
    # # # # CV测试 # # # #
    ####################
    # # 输入文件路径
    # in_file_path=in_folder_path='D:/Project/NER_Interface_v3_20190412/ner_interface/data/CV_Origin_Data'
    # # 输出文件路径
    # out_file_path="D:/Project/NER_Interface_v3_20190412/ner_interface/output/cv/"
    # # 输出文件格式-文件/list
    # out_type='f'
    # # 获取cv实体抽取数据
    # get_cv_data(in_file_path,out_file_path,out_type)

    ####################
    # # # # JD测试 # # # #
    ####################

    # 输入文件路径
    in_file_path='D:/Project/NER_Interface_v3_20190412/ner_interface/data/JD_Origin_Data/RocheTwoPositions.xls'
    # 输出文件路径
    out_file_path = "D:/Project/NER_Interface_v3_20190412/ner_interface/output/jd/"
    # 输出文件格式-文件/list
    out_type = 'f'
    # 获取jd实体抽取数据
    get_jd_data(in_file_path,out_file_path,out_type)
