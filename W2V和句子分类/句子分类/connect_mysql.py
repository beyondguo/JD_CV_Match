# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 19:01:28 2018

@author: x1c
"""

import MySQLdb
db = MySQLdb.connect(host='118.190.77.33',user='root',passwd='ShufeSime!23456',db='sf_ls',charset='utf8')

cursor = db.cursor()

select_sql = "SELECT * FROM sf_jd WHERE CATEGORY='经营/战略'"
cursor.execute(select_sql)
result = cursor.fetchall()

sf_op_req_f = open('sf_op_req.txt','w',encoding='utf-8')
for each in result:
    if each[3]:
        sf_op_req_f.write(each[3])
        print('.',end='')



        
        
    
    
    
