# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 06:57:35 2017

@author: apple
"""


# Usage:
# $ source activate mlenv
# $ python EasyDepth.py

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import layers
import numpy as np
from keras import backend as K
import random

data_folder = '../SinusoidSinusoid3/SinusoidSinusoid3/'
# 1.
# data_file_name = 'SinusoidSinusoid3_18_1.csv'
# 2.
#data_file_name = 'SinusoidSinusoid3_18_1.csv'
# 3.
# data_file_name = 'SinusoidSinusoid3_18_50.csv'
# 4.
# data_file_name = 'SinusoidSinusoid3_19_1.csv'
# 5.
# data_file_name = 'SinusoidSinusoid3_19_10.csv'
# 6.
# data_file_name = 'SinusoidSinusoid3_19_50.csv'
# 7.
data_file_name = 'SinusoidSinusoid3_rand_rand.csv'

def read_data(file):
    data, label = [], []
    with open(data_folder + file) as f:
        for line in f:
            line = line.strip('\n')
            line_split = line.split(',')
            data.append([float(line_split[1]), float(line_split[2]), float(line_split[3])])
            label.append([float(line_split[0])])
    return np.array(data), np.array(label)


model = Sequential()

#model.add(Dense(units=12, input_dim=3, activation='sigmoid'))

model.add(Dense(units= 12, input_dim=3, activation='sigmoid'))
model.add(Dense(units=12))
model.add(layers.advanced_activations.LeakyReLU(alpha=0.3))
#model.add(Dense(units=12, activation='relu'))
model.add(Dense(units=1, activation='relu'))

model.compile(optimizer='sgd', loss='mean_absolute_error')


data, label = read_data(data_file_name)
test_data, test_label = data[:10000], label[:10000]
tune_data, tune_label = data[10000:20000], label[10000:20000]
train_data, train_label = data[20000:], label[20000:]

model.fit(train_data, train_label, epochs=500, shuffle = True, batch_size=128)

loss_and_metrics = model.evaluate(test_data, test_label, batch_size=128)
print ('')
print (loss_and_metrics)

# predicts = model.predict(test_data, batch_size=128)

# for _ in range(1, 100):
#     index = random.randint(0, len(test_label))
#     print test_label[index], predicts[index]