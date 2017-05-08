# Usage:
# $ source activate mlenv
# $ python EasyDepth.py

from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from keras import backend as K
import random
from keras import layers
import keras

data_folder = '../Datasets/SinusoidSinusoid3/'
# 1.
data_file_names = ['SinusoidSinusoid3_18_1.csv',
                    # 2.
                    'SinusoidSinusoid3_18_10.csv',
                    # 3.
                    'SinusoidSinusoid3_18_50.csv',
                    # 4.
                    'SinusoidSinusoid3_19_1.csv',
                    # 5.
                    'SinusoidSinusoid3_19_10.csv',
                    # 6.
                    'SinusoidSinusoid3_19_50.csv',
                    # 7.
                    'SinusoidSinusoid3_rand_rand.csv']

data_file_name = data_file_names[7 - 1];

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

model.add(Dense(units=12, input_dim=3, activation='sigmoid'))
# model.add(Dense(units=12, activation=layers.advanced_activations.LeakyReLU(alpha=0.3)))
# model.add(Dense(units=1, activation=layers.advanced_activations.LeakyReLU(alpha=0.3)))
model.add(Dense(units=1, activation='sigmoid'))

sgd = keras.optimizers.SGD(lr=0.1)

model.compile(optimizer=sgd, loss='mean_absolute_error')


data, label = read_data(data_file_name)
test_data, test_label = data[:10000], (label[:10000]/10000)
tune_data, tune_label = data[10000:20000], (label[10000:20000]/10000)
train_data, train_label = data[20000:], (label[20000:]/10000)


model.fit(train_data, train_label, epochs=500, batch_size=128)

loss_and_metrics = model.evaluate(test_data, test_label, batch_size=128)

print ''
print loss_and_metrics

# predicts = model.predict(test_data, batch_size=128)

# for _ in range(1, 100):
#     index = random.randint(0, len(test_label))
#     print test_label[index], predicts[index]

for i in range(7):
    data_file_name = data_file_names[i]
    data, label = read_data(data_file_name)
    test_data, test_label = data[:10000], label[:10000]
    loss_and_metrics = model.evaluate(test_data, test_label, batch_size=128)
    print ''
    print 'dataset ' + str(i + 1) + ': ' + str(loss_and_metrics)

