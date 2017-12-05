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
X = dataset[:,1:4]
Y = dataset[:,0]

def customF(x):
	norm = tf.random_normal([12], mean=-1, stddev=x)
	print norm
	return norm

def custom_shape(input_shape):
	return input_shape

# create model
model = Sequential()
model.add(Dense(12, input_dim=3, activation='sigmoid'))
model.add(Dense(12, activation='relu'))
model.add(Lambda(customF, output_shape = custom_shape))
model.add(Dense(1, activation='relu'))


#convout1_f = tf.function([model.get_input(train=True)], convout1.get_output(train=True))
#print convout1_f
#inp = model.input                                           # input placeholder
#outputs = [layer.output for layer in model.layers]          # all layer outputs
#functors = [K.function([inp], [out]) for out in outputs]i

# Compile model
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])


#print outputs[1]
model.fit(X, Y, epochs=10000, batch_size=10)


#scores = model.evaluate(X, Y)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# calculate predictions
#predictions = model.predict(X)
# round predictions
#rounded = [round(x[0]) for x in predictions]
#print(rounded)

