#import os
#import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import math
import time
from keras.models import load_model

import pymysql

while True:
    conn = pymysql.connect(host='localhost', user='ROOT', password=' ', db='solar', charset='utf8')
    curs = conn.cursor()

    sql = 'select gyro_x, gyro_y from solar order by num desc limit 4'
    curs.execute(sql)

    rows = curs.fetchall()
    print(rows)

    conn.close()

    data_x = np.array([rows[i][0] for i in range(4)])
    data_y = np.array([rows[i][1] for i in range(4)])
    data_x.astype('float32')
    data_y.astype('float32')

    print('===== data load =====')

    print(data_x)
    print(data_y)

    test_x = np.reshape(data_x, (len(data_x),1))
    test_y = np.reshape(data_y, (len(data_y),1))


    # normalizations
    scaler = MinMaxScaler(feature_range=(0, 1))
    nptf_x = scaler.fit_transform(test_x)
    nptf_y = scaler.fit_transform(test_y)

    #model.save('my_model.h5')
    model = load_model('my_model.h5')

    # predict last value (or tomorrow?)
    last_x = nptf_x[-1]
    last_x = np.reshape(last_x, (1, 1, 1))

    last_y = nptf_y[-1]
    last_y = np.reshape(last_y, (1, 1, 1))

    # lastX 를 넣으면된다. 데이터 예측 할때 

    print(last_x)

    lastY = model.predict(last_x)
    lastY2 = np.reshape(lastY, (1, 1, 1))

    lastY2 = model.predict(lastY2)
    lastY3 = np.reshape(lastY2, (1, 1, 1))

    lastY3 = model.predict(lastY3)
    lastY4 = np.reshape(lastY3, (1, 1, 1))

    lastY4 = model.predict(lastY4)
    lastY5 = np.reshape(lastY4, (1, 1, 1))


    lastY = scaler.inverse_transform(lastY)
    lastY2 = scaler.inverse_transform(lastY2)
    lastY3 = scaler.inverse_transform(lastY3)
    lastY4 = scaler.inverse_transform(lastY4)

    print('Predict gyro_x first: %d' % lastY)  
    print('Predict gyro_x second: %d' % lastY2)  
    print('Predict gyro_x third: %d' % lastY3)  
    print('Predict gyro_x fourth: %d' % lastY4)  


    print(last_y)

    lastY = model.predict(last_y)
    lastY2 = np.reshape(lastY, (1, 1, 1))

    lastY2 = model.predict(lastY2)
    lastY3 = np.reshape(lastY2, (1, 1, 1))

    lastY3 = model.predict(lastY3)
    lastY4 = np.reshape(lastY3, (1, 1, 1))

    lastY4 = model.predict(lastY4)
    lastY5 = np.reshape(lastY4, (1, 1, 1))


    lastY = scaler.inverse_transform(lastY)
    lastY2 = scaler.inverse_transform(lastY2)
    lastY3 = scaler.inverse_transform(lastY3)
    lastY4 = scaler.inverse_transform(lastY4)

    print('Predict gyro_y first: %d' % lastY)  
    print('Predict gyro_y second: %d' % lastY2)  
    print('Predict gyro_y third: %d' % lastY3)  
    print('Predict gyro_y fourth: %d' % lastY4)  

    time.sleep(5)