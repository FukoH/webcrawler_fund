#! usr/bin/python
# -*- coding: utf-8 -*-

from database import DbClient
import pandas as pd
from dateutil.parser import parse
import numpy as np
import argparse

import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import crawler

def load_data_by_code(code):
    '''
    
    :param code: 基金代码
    :return: 时间序列数据,基金名称
    '''
    client = DbClient()
    doc = client.find_by_code(code)
    his_list = doc['data']  # a list of dict
    time_series = pd.DataFrame([d['accumulatedNet'] for d in his_list],
                               index=[parse(d['period']) for d in his_list],
                               columns=['net'])
    # print(type(time_series.index))
    # print(time_series.head(10))519163
    name = client.find_name_by_code(code)
    return time_series, name


def load_data_by_code_online(code):
    '''

    :param code: 基金代码
    :return: 时间序列数据,基金名称
    '''
    doc = crawler.get_foundation_by_code_online(code)
    his_list = doc['data']  # a list of dict
    time_series = pd.DataFrame([d['accumulatedNet'] for d in his_list],
                               index=[parse(d['period']) for d in his_list],
                               columns=['net'])
    # print(type(time_series.index))
    # print(time_series.head(10))519163
    return time_series

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


parser = argparse.ArgumentParser(description='Predict foundation accumulate net value')
parser.add_argument('fcode', type=str,
                    help='foundation id to predict')
args = parser.parse_args()

f_code = args.fcode
dataset = load_data_by_code_online(f_code)
dataset = dataset.sort_index(ascending=True)
# 标准化数据
scaler = MinMaxScaler(feature_range=(0, 1))
time = dataset.index.sort_values(ascending=True)
dataset = dataset['net'].dropna(axis=0)
dataset.astype('float64')
dataset = np.asarray(dataset).reshape(len(dataset), 1)
dataset = scaler.fit_transform(dataset)

"""
分隔训练集和测试集合
"""
# split into train and test sets
size_a = 0.67
# from set_train_size import size_a
# from set_train_size import size_b

train_size = int(len(dataset) * size_a)
test_size = len(dataset) - train_size
# test_size = int(len(dataset) *(1-size_b))
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
# train, test = dataset[0:train_size,:], dataset[test_size:,:]

"""
将数据转换成模型需要的形状，X=t and Y=t+1
"""
look_back = 1
# from set_super_parameter import look_back

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
"""
将数据转换成模型需要的形状，[样本samples,时间步 time steps, 特征features]
"""
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

"""
搭建LSTM神经网络
"""
sample = 8
nb_epoch = 50
optimizer = 'adam'
# from set_super_parameter import sample
# from set_super_parameter import nb_epoch
# from set_super_parameter import optimizer

model = Sequential()
model.add(LSTM(sample, input_dim=look_back))
# model.add(layers.Dropout(0.01))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer=optimizer)
model.fit(trainX, trainY, nb_epoch=nb_epoch, batch_size=1, verbose=2)

"""
预测数据，对训练集和测试集上的数据进行预测
"""
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
"""
把归一化的预测数据，转换成业务数据的范围和格式
"""
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
"""
模型评估，计算均方根误差RMSE( root mean squared error)
"""
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))

"""
把训练集合上的预测结果做成图
"""
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
"""
把测试集合上的预测结果做成图
"""
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
"""
把原始数据作为基线做成图
"""
plt.plot(time,scaler.inverse_transform(dataset),'b-',label='true')
plt.plot(time,trainPredictPlot, 'r--',label='train')
plt.plot(time,testPredictPlot, 'g--',label='test')
plt.legend()
plt.title('Accumulated Net Value Prediction for %s Based On Past 200 Days' % (f_code))
plt.savefig('test.png')
plt.show()
