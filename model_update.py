## 하루에 한번 모델 업데이트 

import threading
import time
import os
from sklearn import tree
import pymysql
import numpy as np
import pickle
import graphviz

end = False

def execute_func_create(second=1):
    global end
    if end:
        return
    
    #TODO
    conn = pymysql.connect(host='localhost', user='root', password='2580',
                      db='solar2', charset='utf8')
    try:
        with conn.cursor() as curs:
            curs = conn.cursor()

            sql = 'select temperature, gyro_x, gyro_y, gyro_z from water'
            curs.execute(sql)

            rows = curs.fetchall()
            
            target = []
            
            for i in range(len(rows)):
                if int(rows[i][0]) > 80:
                    target.append(1)
                elif int(rows[i][1]) < -10 or int(rows[i][1]) > 10:
                    target.append(2)
                elif int(rows[i][2]) < -10 or int(rows[i][2]) > 10:
                    target.append(3)
                elif int(rows[i][3]) < -10 or int(rows[i][3]) > 10:
                    target.append(4)   
                else:
                    target.append(0)

            train_data = np.array(rows)
            target = np.array(target)

            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(train_data, target)

            now = time.localtime()
            #now_time = "%04d-%02d-%02d_%02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
            now_time = "%04d-%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday)

            ROOT_DIR = os.getcwd()

            MODEL_DIR = os.path.join(ROOT_DIR)
            
            
            list_pickle_path = MODEL_DIR+'/log/'+now_time+'.pkl'
            list_pickle = open(list_pickle_path, 'wb')
            pickle.dump(clf, list_pickle)
            list_pickle.close()

            feature_name = np.array(['temperature', 'X','Y','Z'])
            class_name = np.array(['safe', 'fire', 'X_Fallen','Y_Fallen','Z_Fallen'])

            dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=feature_name,  
                         class_names=class_name,  
                         filled=True, rounded=True,  
                         special_characters=True)
            graph = graphviz.Source(dot_data)
            graph.render("graph - "+now_time)


            print('model update ' + now_time)
    finally:
        conn.close()
    
    threading.Timer(second, execute_func_create,[second]).start()
        
execute_func_create(86400)