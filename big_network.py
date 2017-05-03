from keras.layers import Dense
from keras.layers import GaussianNoise
from keras.layers import Lambda
from keras import backend as K
from keras.models import Sequential
import tensorflow as tf
import numpy as np

#import sys
#sys.path.insert(0, 'Users/nikhil/Code/keras/keras/layers')
#sys.path.insert(0, 'Users/nikhil/Code/keras/keras/')
#import layers.core as core
#import keras.models as models
#import keras.keras.layers.noise as noise

np.random.seed(7)
# load dataset
dataset = np.loadtxt("/Users/nikhil/data/Sinusoid3/Sinusoid3_200.csv", delimiter=",")
# split into input (X) and output (Y) variables
L1_test = dataset[:10000,1:200]
L2_test = dataset[:10000,201:400]
L3_test = dataset[:10000,401:600]
input_array_test =  np.array([L1_test.T, L2_test.T, L3_test.T])
input_array_test = input_array_test.T
label_test = dataset[:10000, 0]

L1_tune = dataset[10000:20000,1:200]
L2_tune = dataset[10000:20000,201:400]
L3_tune = dataset[10000:20000,401:600]
input_array_tune =  np.array([L1_tune.T, L2_tune.T, L3_tune.T])
input_array_tune = input_array_tune.T
label_tune = dataset[10000:20000, 0]

L1_train = dataset[20000:,1:200]
L2_train = dataset[20000:,201:400]
L3_train = dataset[20000:,401:600]

input_array_train =  np.array([L1_train.T, L2_train.T, L3_train.T])
input_array_train = input_array_train.T


label_train = dataset[20000:, 0]

def customF(x):
	norm = tf.random_normal([12], mean=-1, stddev=x)
	print norm
	return norm

def custom_shape(input_shape):
	return input_shape

input_l = Input(shape=(200, 3))
# create model
model = Sequential()
model.add(Dense(12, input_l, activation='sigmoid'))
model.add(Dense(12, activation='relu'))
model.add(Lambda(customF, output_shape = custom_shape))
model.add(Dense(1, activation='relu'))



# Split input to 2 streams
#a, b = Lambda(lambda x: x[:, :10], output_shape=(10,))(input),  Lambda(lambda x: x[:, 10:], output_shape=(10,))(input)
# One stream goes through the hidden layer
#c = Dense(10)(a)
# Other stream goes directly to output
#output = merge([c, b], mode='concat')
#model = Model(input, output)




#convout1_f = tf.function([model.get_input(train=True)], convout1.get_output(train=True))
#print convout1_f
#inp = model.input                                           # input placeholder
#outputs = [layer.output for layer in model.layers]          # all layer outputs
#functors = [K.function([inp], [out]) for out in outputs]i

# Compile model
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])


#print outputs[1]
model.fit(input_array_train, label_train, epochs=10000, batch_size=10)
