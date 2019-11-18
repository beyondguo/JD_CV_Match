#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/14 10:31
# @Author  : Frances
# @Site    : 
# @File    : data_pre_handle.py
# @Software: PyCharm
import os
import re
import time
import json
import  datetime
import pandas as pd

###################
# # # #数据前处理模块
###################

# 文件输入数据处理模块

# # JD输入数据前处理
def  jd_data_pre_handle(in_folder_path):
    '''

    :param in_folder_path: 读入路径下的数据
    :return: 读取的数据
    '''
    # 检查原始数据所在文件夹
    print(in_folder_path)
    assert os.path.exists(in_folder_path), "路径不存在，重新输入！"

    # 计时
    start = time.clock()
    resume_num = 0

    # 实体抽取
    jd_list = abstract_infor_from_jd(in_folder_path)

    # 程序运行时间
    elapsed = (time.clock() - start)

    # 返回list
    return jd_list

# 从JD中提取信息
def abstract_infor_from_jd(in_folder_path):
    jd_tojson(in_folder_path)
    with open(os.path.join('./log/', 'alljd.json'), "r", encoding='utf8') as fp:
        jd_json = json.load(fp)
    all_jd = []
    jd = {}
    for position in jd_json:
        jd['id'] = position['position_id']
        jd['name'] = position['name']
        jd['职能'] = position['str_job_child_type'].split('/')
        jd['学历'] = position['str_degree']
        jd['年限'] = position['str_work_years']
        jd['省'] = position['str_province']
        jd['市'] = position['str_city']
        jd['专业'] = ''
        jd['技能'] = ''
        jd['业务'] = ''
        jd['行业'] = ''
        jd['产品'] = ''
        jd['详细信息'] =position['detail']
        all_jd.append(jd)
        jd = {}
    return all_jd

# 将excel格式的jd转化为json形式，并去掉detail里面的英文说明
def jd_tojson(in_folder_path):
    df = pd.read_excel(in_folder_path)
    chinese_detail = ''
    for idex,row in df.iterrows():
        chinese_list = re.findall('([^\x00-\xff][^<]+)[。<]',row['detail'])
        for i in chinese_list:
            chinese_detail += i+'\n'
        df.iloc[idex,2] = chinese_detail.strip() #2为detail所在的列数
        chinese_detail = ''

    newfile = open(os.path.join('./log/','alljd.json'), "w", encoding='utf8')
    asjson = df.to_json(orient='records')
    asjson = json.loads(asjson)
    newasjson = json.dumps(asjson, ensure_ascii=False, indent=4)
    newfile.write(newasjson)
    newfile.close()

# # CV输入数据前处理
def  cv_data_pre_handle(in_folder_path):
    '''

    :param in_folder_path: 读入路径下的数据
    :return: 读取的数据
    '''
    # 检查原始数据所在文件夹
    print(in_folder_path)
    assert os.path.exists(in_folder_path),"路径不存在，重新输入！"

    # 计时
    start = time.clock()
    resume_num = 0

    pre_handled_data = []

    # 获取文件夹下的所有文件夹
    subfs_crude = [subf for subf in os.listdir(in_folder_path) if
                   os.path.isdir(os.path.join(in_folder_path, subf))]

    # 提取实体抽取字段
    for subf in subfs_crude:
        # 原始数据
        cv_json_folder = os.path.join(in_folder_path, subf)

        # 返回所有数据
        pre_handled_data.append(abstract_information_from_cv(cv_json_folder))

    # 返回读入数据
    return pre_handled_data

# 抽取CV中信息
def abstract_information_from_cv(cv_json_folder):
    all_cv = []
    cv = {}
    for root, dirs, files in os.walk(cv_json_folder):
        for file in files:
            # 读CV文件
            with open(root + '\\' + file, encoding='utf8') as fp:
                json_text = json.load(fp)

            # 当前目录
            pathorigin = str(root + '\\' + file)
            # 保证路径中特殊字符不被转义
            pathorigin = pathorigin.replace("\\", '/')
            # 保存在字段中
            cv['in_path'] = pathorigin
            # 保存cv的id
            cv['id'] = os.path.splitext(file)[0]

            # 保存其他字段

            # 工作年限
            if 'workYears' in json_text.keys():
                cv['工作年限'] = json_text['workYears']
            else:
                cv['工作年限'] = ''

            # 目标职位工作城市
            if 'jobObjective' in json_text.keys():
                cv['工作城市'] = []
                city = []
                if json_text['jobObjective'].get('city'):
                    obj = json_text['jobObjective']
                    if re.findall(';', str(obj['city'])):
                        city = str(obj['city']).split(';')
                    elif re.findall(',', str(obj['city'])):
                        city = str(obj['city']).split(';')
                    else:
                        city.append(str(obj['city']).strip())

                    cv['工作城市'] = city

            # 教育背景
            if 'eduExp' in json_text.keys() and len(json_text['eduExp']):
                degree = []
                school = []
                major = []
                for edu in json_text['eduExp']:
                    # 学历
                    if edu.get('degree'):
                        degree.append(edu['degree'])
                        # 学校
                        if edu.get('school'):
                            school.append(edu['school'])
                    # 专业
                    if edu.get('major'):
                        major.append(edu['major'])

                cv['专业'] = major
                cv['毕业院校'] = school
                cv['学历'] = degree

            # 工作经验
            if 'workExp' in json_text.keys() and len(json_text['workExp']):
                workExp = []
                for wo in json_text['workExp']:
                    work = {}
                    # 工作公司
                    if wo.get('company'):
                        work['工作公司'] = wo['company']
                    # 公司规模
                    if wo.get('scale'):
                        work['公司规模'] = wo['scale']
                    # 行业
                    if wo.get('industry'):
                        work['工作行业'] = wo['industry'].split('/')
                    # 部门
                    if wo.get('department'):
                        work['工作部门'] = wo['department'].split('/')
                    # 职位
                    if wo.get('position'):
                        work['工作职位'] = wo['position'].split('/')
                    # 当前岗位工作年限
                    if wo.get('startDate') and re.findall(r'\d', wo['startDate']):
                        starttime = get_time(wo['startDate'])
                        if (wo.get('endDate')) and re.findall(r'\d', wo['endDate']):
                            endtime = get_time(wo['endDate'])
                            delta = round((endtime - starttime).days / 365, 1)
                            if delta < 20:
                                work['岗位工作年限'] = delta
                            else:
                                newdelta = get_time(time.strftime("%Y-%m")) - starttime
                                work['岗位工作年限'] = round(newdelta.days / 365, 1)
                        else:
                            newdelta = get_time(time.strftime("%Y-%m")) - starttime
                            work['岗位工作年限'] = round(newdelta.days / 365, 1)

                    # 具体描述【非结构抽取】
                    if wo.get('detail'):
                        work['工作描述'] = wo['detail']
                    workExp.append(work)

                cv['工作经验'] = workExp

            # 项目经验
            if 'projectExp' in json_text.keys() and len(json_text['projectExp']):
                projectExp = []
                for pro in json_text['projectExp']:
                    project = {}
                    # 项目名称
                    if pro.get('name'):
                        project['项目名称'] = pro.get('name')
                    # 项目时长
                    if pro.get('startDate') and re.findall(r'\d', pro['startDate']):
                        starttime = get_time(pro['startDate'])
                        if (pro.get('endDate')) and re.findall(r'\d', pro['startDate']):
                            endtime = get_time(pro['endDate'])
                            delta = round((endtime - starttime).days / 365, 1)
                            if delta < 20:
                                project['项目时长'] = delta
                            else:
                                newdelta = get_time(time.strftime("%Y-%m")) - starttime
                                project['项目时长'] = round(newdelta.days / 365, 1)
                        else:
                            newdelta = get_time(time.strftime("%Y-%m")) - starttime
                            project['项目时长'] = round(newdelta.days / 365, 1)

                    # 具体描述【非结构抽取】
                    if pro.get('detail'):
                        project['项目描述'] = pro['detail']
                    projectExp.append(project)

                cv['项目经验'] = projectExp

            # 证书抽取为技能
            if 'certificate' in json_text.keys():
                if len(json_text['certificate']):
                    certificate = re.findall('[0-9]{4}[/.-][0-9]{1,2}[/.-]*[0-9]*[0-9]* *([\u4e00-\u9fa5a-zA-Z0-9]+)',
                                             json_text['certificate'])
                    cv['技能'] = certificate

            # 语言
            if 'langAbility' in json_text.keys() and len(json_text['langAbility']):
                lang1 = json_text['langAbility'].split('\n')
                lang2 = []
                cv['语言'] = {}

                for item in lang1:  # 去掉空项
                    item = item.strip()
                    if item:
                        lang2.append(item)

                for item in lang2:
                    if len(item.split(' ')) == 2:
                        cv['语言'][item.split(' ')[0]] = item.split(' ')[1]
                    elif len(item.split(' ')) == 1:
                        cv['语言'][item.split(' ')[0]] = ''

            # 自我评价【非结构抽取】
            if 'selfComm' in json_text.keys() and len(json_text['selfComm']):
                cv['自我评价'] = json_text['selfComm']

            # 业务
            cv['业务'] = ''
            # 产品
            cv['产品'] = ''
            all_cv.append(cv)
            cv = {}
        return all_cv

# 处理时间格式
def get_time(time):
    style1  = '-'
    style2 = '/'
    get_time = re.findall(r'(\d{4}[-/]\d{1,2})[-/]*\d{0,2}',time)
    if get_time:
        if style1 in get_time[0]:
            newtime = datetime.datetime.strptime(get_time[0], '%Y-%m')
            return newtime
        elif style2 in get_time[0]:
            newtime = datetime.datetime.strptime(get_time[0], '%Y/%m')
            return newtime
        else:
            print('有新的日期格式：',time)

if __name__=="__main__":
    # # TODO:文件路径检查
    # in_folder_path='D:/Project/NER_Interface_v3_20190412/ner_interface/data/CV_Origin_Data'
    # ll=cv_data_pre_handle(in_folder_path)
    in_folder_path='D:/Project/NER_Interface_v3_20190412/ner_interface/data/JD_Origin_Data/RocheTwoPositions.xls'
    jd_data_pre_handle(in_folder_path)