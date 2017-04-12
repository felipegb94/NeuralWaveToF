# Usage:
# $ source activate mlenv
# $ python EasyDepth.py

from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from keras import backend as K


data_folder = '../Datasets/'
train_data_file = data_folder + 'EasyDepthData_Train.csv'
tune_data_file  = data_folder + 'EasyDepthData_Tune.csv'
test_data_file  = data_folder + 'EasyDepthData_Test.csv'


def read_data(file):
    data, label = [], []
    with open(file) as f:
        for line in f:
            line = line.strip('\n')
            line_split = line.split(',')
            data.append([float(line_split[1]), float(line_split[2]), float(line_split[3])])
            label.append([float(line_split[0])])
    return np.array(data), np.array(label)



model = Sequential()

model.add(Dense(units=64, input_dim=3, activation='relu'))
model.add(Dense(units=1, activation='relu'))

model.compile(optimizer='sgd', loss='mean_absolute_error')


train_data, train_label = read_data(train_data_file)
tune_data,  tune_label  = read_data(tune_data_file)
test_data,  test_label  = read_data(test_data_file)

model.fit(train_data, train_label, epochs=500, batch_size=128)

loss_and_metrics = model.evaluate(tune_data, tune_label, batch_size=128)

print ''
print loss_and_metrics