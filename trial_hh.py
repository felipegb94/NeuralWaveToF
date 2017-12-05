# Usage:
# $ source activate mlenv

from keras.models import Sequential
from keras.layers import Dense, Activation, LocallyConnected1D, Flatten
import numpy as np
from keras import backend as K
import random
from keras import layers

data_folder = '/Users/nikhil/data/Sinusoid3/'
# 1.
data_file_names = ['Sinusoid3_200.csv']

data_file_name = data_file_names[0];

def read_data(file):
    data, label = [], []
    with open(data_folder + file) as f:
        for line in f:
            line = line.strip('\n')
            line_split = line.split(',')
            tmp = []
            for i in range(1, len(line_split)):
                tmp.append([float(line_split[i])])
            data.append(tmp)
            label.append([float(line_split[0])])
    return np.array(data), np.array(label)

def custom_activation(x):
    # return K.sigmoid(x + K.random_normal_variable((K.ndim(x), 1), 0, x))
    # noise = K.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    noise = K.random_normal(shape=K.shape(x), mean=0.0, stddev=10, dtype=K.dtype(x))
#    return K.relu(x + noise)
    return K.sigmoid(x)

model = Sequential()

model.add(LocallyConnected1D(1, 200, strides=200, input_shape=(600, 1)))
#model.add(Flatten())
#model.add(Dense(units=12, activation='sigmoid'))
model.add(Dense(units=12, activation=custom_activation))
#model.add(Dense(units=12, activation=layers.advanced_activations.LeakyReLU(alpha=0.3)))
model.add(Dense(units=1, activation=layers.advanced_activations.LeakyReLU(alpha=0.3)))

model.compile(optimizer='sgd', loss='mean_absolute_error')


data, label = read_data(data_file_name)
test_data, test_label = data[:10000], label[:10000]
tune_data, tune_label = data[10000:20000], label[10000:20000]
train_data, train_label = data[20000:], label[20000:]

model.fit(train_data, train_label, epochs=10000, batch_size=64)

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

