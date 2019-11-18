#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/14 10:34
# @Author  : Frances
# @Site    : 
# @File    : data_post_handle.py
# @Software: PyCharm

import os
import json

###################
# # # #后处理模块
###################

# 文件输出数据处理模块

# # 默认返回列表
def  data_output_handle(entity_result, out_file_path, out_type=0):
    # 检查输出路径
    assert os.path.exists(out_file_path),"输出路径不存在！"

    # 输出到指定路径
    if out_type!=0:
        assert os.path.exists(out_file_path),"输出路径不存在！"
        for folder in entity_result:
            if type(folder)==dict:
                out_filename = str(out_file_path) + str(folder['name'] + '_' + str(folder['id']) + '_entity.json')
                out_path = os.path.join(out_file_path, out_filename)
                json_str = json.dumps(folder, ensure_ascii=False, indent=4)  # 注意这个indent参数(保留格式)
                json_file = open(out_path, 'w', encoding='utf-8')
                json_file.write(json_str)
                json_file.close()
                print("抽取成功，存储路径为:{}".format(out_file_path))
            else:
                for files in folder:
                    out_filename = str(out_file_path)+str(files['name']+'_'+files['id']+ '_entity.json')
                    out_path = os.path.join(out_file_path, out_filename)
                    json_str = json.dumps(files, ensure_ascii=False, indent=4)  # 注意这个indent参数(保留格式)
                    json_file = open(out_path, 'w', encoding='utf-8')
                    json_file.write(json_str)
                    json_file.close()
                print("抽取成功，存储路径为:{}".format(out_file_path))

    # 默认返回列表
    else:
        print("抽取成功，返回列表")
        return entity_result