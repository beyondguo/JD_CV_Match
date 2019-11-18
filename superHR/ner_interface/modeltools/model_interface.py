#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/9 12:36
# @Author  : Frances
# @Site    : 
# @File    : model_interface.py
# @Software: PyCharm

import os
import tensorflow as tf
import pickle

from modeltools.model import Model
from datatools.utils import get_logger, make_path, clean, create_model, save_model
from datatools.utils import print_config, save_config, load_config, test_ner,load_maps
from datatools.data_utils import load_word2vec, create_input, input_from_line, BatchManager

##################
# # 实体识别内部模块
#################

# 处理CV实体提取
class cvEntity(object):
    def __init__(self):
        # 配置信息路径
        self.ckpt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        self.config_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config_file')
        self.map_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'map', 'maps.pkl')
        self.log_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'log', 'entity_extract.log')

        # 读取配置信息
        self.config = load_config(self.config_file_path)
        self.logger = get_logger(self.log_file_path)
        with open(self.map_file_path, "rb") as f:
            self.char_to_id, self.id_to_char, self.tag_to_id, self.id_to_tag = pickle.load(f)
        f.close()

    # 主要针对Cv里面工作经历detail和项目经历detail以及自我评价进行非结构化抽取
    def entity_extract(self, resume_process):
        # 清除上次的图
        tf.reset_default_graph()
        result = []
        # limit GPU memory
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        graph = tf.Graph()
        with tf.Session(config=tf_config, graph=graph) as sess:
            mymodel = create_model(sess, Model, self.ckpt_path, load_word2vec, self.config, self.id_to_char, self.logger, graph=graph)
            # 循环抽取实体
            results_folder=[]
            # 不同文件夹

            for files in resume_process:
                results_files=[]
                # 不同文件循环
                for file in files:
                    experience={}
                    experience['工作经验']=[]
                    experience['项目经验'] = []
                    if '工作经验' in file.keys() and len(file['工作经验']):
                        # 工作经验循环
                        dicts = {}
                        for workExp in file['工作经验']:
                            # 工作经验
                            dicts['工作描述']=[]
                            if '工作描述' in workExp.keys():
                                # 所有的描述
                                workDes=[]
                                descripts=str(workExp['工作描述']).strip().split('\n')
                                if len(descripts):
                                    # 每个工作经验里多个句子
                                    descripts = [descript for descript in descripts if len(descript) > 0]
                                    for descript in descripts:
                                        description = mymodel.evaluate_txt(sess, input_from_line(descript, self.char_to_id), self.id_to_tag)
                                        workDes.extend(description)

                                    workExp['工作描述']=get_keyword(workDes)
                            experience['工作经验'].append(workExp)
                        file['工作经验'] = experience['工作经验']

                    if '项目经验' in file.keys() and len(file['项目经验']):
                        # 工作经验循环
                        dicts = {}
                        for workExp in file['项目经验']:
                            # 工作经验
                            dicts['项目描述'] = []
                            if '项目描述' in workExp.keys():
                                # 所有的描述
                                workPro = []
                                descripts = str(workExp['项目描述']).strip().split('\n')
                                if len(descripts):
                                    # 每个工作经验里多个句子
                                    descripts = [descript for descript in descripts if len(descript) > 0]
                                    for descript in descripts:
                                        description = mymodel.evaluate_txt(sess,
                                                                           input_from_line(descript, self.char_to_id),
                                                                           self.id_to_tag)
                                        workPro.extend(description)

                                workExp['项目描述'] = get_keyword(workPro)
                            experience['项目经验'].append(workExp)
                        file['项目经验'] = experience['项目经验']

                    if '自我评价' in file.keys() and len(file['自我评价']):
                        dict={}
                        descripts = file['自我评价'].strip().split('\n')
                        # print("自我评价提取之前：",descripts)
                        if len(descripts):
                            descripts = [descript for descript in descripts if len(descript) > 0]
                            workSelf= []
                            for descript in descripts:
                                description = mymodel.evaluate_txt(sess, input_from_line(descript, self.char_to_id),
                                                                    self.id_to_tag)
                                workSelf.extend(description)
                        dicts['自我评价']=get_keyword(workSelf)
                        file['自我评价'] = dicts['自我评价']

                    # 当前文件夹
                    results_files.append(file)
                 # 所以文件夹
                results_folder .append(results_files)
            return results_folder

class jdEntity(object):
    def __init__(self):
        # 配置信息路径
        self.ckpt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        self.config_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config_file')
        self.map_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'map', 'maps.pkl')
        self.log_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'log', 'entity_extract.log')

        # 读取配置信息
        self.config = load_config(self.config_file_path)
        self.logger = get_logger(self.log_file_path)
        with open(self.map_file_path, "rb") as f:
            self.char_to_id, self.id_to_char, self.tag_to_id, self.id_to_tag = pickle.load(f)
        f.close()

    def entity_extract(self,jd_process):
        # 清除上次的图
        tf.reset_default_graph()
        result = []
        # limit GPU memory
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        graph = tf.Graph()
        with tf.Session(config=tf_config, graph=graph) as sess:
            mymodel = create_model(sess, Model, self.ckpt_path, load_word2vec, self.config, self.id_to_char,
                                   self.logger, graph=graph)

            # 循环抽取实体
            jd_result = []
            # 不同文件夹

        for jds in jd_process:
            jd_result = []
            if '详细信息' in jds.keys():
                dicts={}
                dicts['详细信息']=[]
                descripts = jds['详细信息'].strip().split('\n')
                if len(descripts):
                    descripts = [descript for descript in descripts if len(descript) > 0]
                    jd_detail = []
                    for descript in descripts:
                        description = mymodel.evaluate_txt(sess, input_from_line(descript, self.char_to_id),
                                                   self.id_to_tag)
                        jd_detail.extend(description)

                    dicts['详细信息'] = get_keyword(jd_detail)
                jds['详细信息'] = dicts['详细信息']
                jd_result.append(jds)
            else:
                jd_result.append(jds)
            return jd_result




# 处理调用模型之后的数据
# 注意：处理后的实体未去重
def get_keyword(resume_result):
    dict = {}
    skill = []
    education = []
    major = []
    industry = []
    position = []
    year = []
    business = []
    products = []
    for se in resume_result:
        if 'value' in se.keys():
            word = se['value']
            if (se['type'] == '技能'):
                skill.append(word)
            elif (se['type'] == '学历'):
                education.append(word)
            elif (se['type'] == '专业'):
                major.append(word)
            elif (se['type'] == '行业'):
                industry.append(word)
            elif (se['type'] == '职能'):
                position.append(word)
            elif (se['type'] == '年限'):
                year.append(word)
            elif (se['type'] == '业务'):
                business.append(word)
            elif (se['type'] == '产品'):
                products.append(word)
    dict['技能'] = skill
    dict['学历'] = education
    dict['专业'] = major
    dict['行业'] = industry
    dict['职能'] = position
    dict['年限'] = year
    dict['业务'] = business
    dict['产品'] = products
    return dict
