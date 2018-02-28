# 안전진단 하기 

import threading
import pymysql
import numpy as np
from sklearn import tree
import pickle
import os
import time

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR,'log')

print(MODEL_DIR)

import stat, glob

def selectLatest():
    os.chdir(MODEL_DIR)
    fileList = glob.glob('*')

    latesMtime = 0
    latesFileName = ''
    for file in fileList:
        mtime = os.stat(file)[stat.ST_MTIME]
        if mtime > latesMtime:
            latestMtime = mtime
            latesFileName = file

    return latesFileName




end = False

def execute_func(second=1):
    global end
    if end:
        return
    
    list_pickle_path = selectLatest()
    list_unpickle = open(list_pickle_path, 'rb')
    clf = pickle.load(list_unpickle)


    #TODO
    conn = pymysql.connect(host='localhost', user='root', password='2580',
                      db='solar2', charset='utf8')
    try:
        with conn.cursor() as curs:
            sql = 'select temperature, gyro_x, gyro_y, gyro_z from water order by num desc limit 2'
            curs.execute(sql)
            rows = curs.fetchall()

            print(rows)

            if int(rows[0][1]) - int(rows[1][1]) > 5 or int(rows[0][1]) - int(rows[1][1]) < -5:
                prediction = '2'
            elif int(rows[0][2]) - int(rows[1][2]) > 5 or int(rows[0][2]) - int(rows[1][2]) < -5:
                prediction = '3'
            elif int(rows[0][3]) - int(rows[1][3]) > 5 or int(rows[0][3]) - int(rows[1][3]) < -5:
                prediction = '4'
            else:
                rows_0 = np.zeros((1,len(rows[0])))
                rows_0[0] = rows[0]
                
                predict_data = np.array(rows_0)
                prediction = clf.predict(predict_data)
            
        with conn.cursor() as curs:
            sql = 'insert into water_predict (state) values (%s)'
            if int(prediction) == 0:
                curs.execute(sql, 'safe')
            elif int(prediction) == 1:

                curs.execute(sql, 'fire')
            elif int(prediction) == 2:
                curs.execute(sql, 'x-collapse')
            elif int(prediction) == 3:
                curs.execute(sql, 'y-collapse')
            elif int(prediction) == 4:
                curs.execute(sql, 'z-collapse')
            
        conn.commit()

        now = time.localtime()
        now_time = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

        print('commit! - '+ now_time)
    finally:
        conn.close()

    threading.Timer(second, execute_func, [second]).start()

execute_func(15)