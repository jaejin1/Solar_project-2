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

    sql = 'select gyro_x, gyro_y from solar1 order by num desc limit 4'
    curs.execute(sql)

    rows = curs.fetchall()
    print(rows)  # db result 

    

    data_x = np.array([rows[i][0] for i in range(4)])
    data_y = np.array([rows[i][1] for i in range(4)])
    #data_z = np.array([rows[i][2] for i in range(4)])
    data_x.astype('float32')
    data_y.astype('float32')
    #data_z.astype('float32')

    print('===== data load =====')

    print(data_x)
    print(data_y)
    #print(data_z)

    test_x = np.reshape(data_x, (len(data_x),1))
    test_y = np.reshape(data_y, (len(data_y),1))
    #test_z = np.reshape(data_z, (len(data_z),1))

    # normalizations
    scaler = MinMaxScaler(feature_range=(0, 1))
    nptf_x = scaler.fit_transform(test_x)
    nptf_y = scaler.fit_transform(test_y)
    #nptf_z = scaler.fit_transform(test_z)

    #model.save('my_model.h5')
    model = load_model('my_model.h5')

    # predict last value (or tomorrow?)
    last_x = nptf_x[-1]
    last_x = np.reshape(last_x, (1, 1, 1))

    last_y = nptf_y[-1]
    last_y = np.reshape(last_y, (1, 1, 1))
    
    #last_z = nptf_z[-1]
    #last_z = np.reshape(last_z, (1, 1, 1))

    print(last_x)

    lastX = model.predict(last_x)
    lastX2 = np.reshape(lastX, (1, 1, 1))

    lastX2 = model.predict(lastX2)
    lastX3 = np.reshape(lastX2, (1, 1, 1))

    lastX3 = model.predict(lastX3)
    lastX4 = np.reshape(lastX3, (1, 1, 1))

    lastX4 = model.predict(lastX4)


    lastX = scaler.inverse_transform(lastX)
    lastX2 = scaler.inverse_transform(lastX2)
    lastX3 = scaler.inverse_transform(lastX3)
    lastX4 = scaler.inverse_transform(lastX4)

    print('Predict gyro_x first: %d' % lastX)  
    print('Predict gyro_x second: %d' % lastX2)  
    print('Predict gyro_x third: %d' % lastX3)  
    print('Predict gyro_x fourth: %d' % lastX4)  

    if lastX4 > 10 or lastX4 < -10:
        sql = 'update solar1 set safe1="1" where num>1 order by num desc limit 1'
        curs.execute(sql)
        print('dangerous!!')
        # add sql

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
    
    if lastY4 > 10 or lastY4 < -10:
        sql = 'update solar1 set safe1="1" where num>1 order by num desc limit 1'
        curs.execute(sql)
        print('dangerous!!')
        # add sql
        
    '''
    print(last_z)

    lastZ = model.predict(last_z)
    lastZ2 = np.reshape(lastZ, (1, 1, 1))

    lastZ2 = model.predict(lastZ2)
    lastZ3 = np.reshape(lastZ2, (1, 1, 1))

    lastZ3 = model.predict(lastZ3)
    lastZ4 = np.reshape(lastZ3, (1, 1, 1))

    lastZ4 = model.predict(lastZ4)


    lastZ = scaler.inverse_transform(lastZ)
    lastZ2 = scaler.inverse_transform(lastZ2)
    lastZ3 = scaler.inverse_transform(lastZ3)
    lastZ4 = scaler.inverse_transform(lastZ4)

    print('Predict gyro_z first: %d' % lastZ)  
    print('Predict gyro_z second: %d' % lastZ2)  
    print('Predict gyro_z third: %d' % lastZ3)  
    print('Predict gyro_z fourth: %d' % lastZ4)  

    if lastZ4 > 10 or lastZ4 < -10:
        sql = 'update solar1 set safe1="1" where num>1 order by num desc limit 1'
        curs.execute(sql)
        print('dangerous!!')
        # add sql
    '''
    
    conn.close()
    time.sleep(5)