# Usage:
# $ source activate mlenv
# $ python EasyDepth.py

# Python Imports
import random
import math
# Library Imports
import keras as krs
# from keras.models import Sequential
# from keras.layers import Dense, Activation
# from keras import layers
# from keras import backend as K
import numpy as np
# Local Imports


def nonoise_activation(x):

    return x

# def custom_activation(x):
#     # return K.sigmoid(x + K.random_normal_variable((K.ndim(x), 1), 0, x))
#     # noise = K.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
#     photonNoise = krs.backend.random_normal(shape=krs.backend.shape(x), mean=0.0, stddev=np.sqrt(np.abs(x)), dtype=krs.backend.dtype(x)) 
#     readNoise = krs.backend.random_normal(shape=krs.backend.shape(x), mean=0.0, stddev=50, dtype=krs.backend.dtype(x)) 
#     print "custom activation!"
#     # return x
#     return x + photonNoise + readNoise
#     # return x + readNoise
    # return K.sigmoid(x)

datasetName = 'Sinusoid3'
datasetPath = '../Datasets/' + datasetName + '/'
numPoints = 200
datasetFilename = datasetName + '_' + str(numPoints) 
k = 3 # Number of modulation functions on dataset

print datasetPath + datasetFilename + '.csv'
dataset = np.loadtxt(datasetPath + datasetFilename + '.csv', delimiter=',')
(numSamples, numCols) = dataset.shape

numTrainSamples = int(math.floor(numSamples * 0.8))
numTestSamples = numSamples - numTrainSamples

trueDepths = dataset[:,0]
trueDepthsTrain = trueDepths[0:numTrainSamples]
trueDepthsTest = trueDepths[numTrainSamples:numSamples]


modulationSamples = np.zeros((numSamples,numPoints,k)) 

for i in np.arange(0,numSamples):
	currSample = dataset[i,1:numCols]
	currSample = currSample.reshape((k,numPoints))
	modulationSamples[i,:,0] = currSample[0,:]
	modulationSamples[i,:,1] = currSample[1,:]
	modulationSamples[i,:,2] = currSample[2,:]

modulationSamplesTrain = modulationSamples[0:numTrainSamples,:,0]
modulationSamplesTest = modulationSamples[numTrainSamples:numSamples,:,0]
print trueDepthsTrain.shape
print trueDepthsTest.shape
print modulationSamplesTrain.shape
print modulationSamplesTest.shape

# # set parameters:
batch_size = 32
epochs = 100
filters = k
kernel_size = (1,numPoints)
input_shape = (1,numPoints,k)


model = krs.models.Sequential()
# TODO: Figure out how to do convolution with Conv1D

varScalingInitializer = krs.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)


model.add(krs.layers.Dense(units=3, input_dim=numPoints, activation=nonoise_activation))

# model.add(krs.layers.Conv2D(filters,
#                  kernel_size,
#                  activation=custom_activation,
#                  use_bias=False,
#                  kernel_constraint=krs.constraints.non_neg(),
#                  # kernel_initializer='random_uniform',
#                  input_shape=input_shape))

# model.add(krs.layers.Dense(units=100, activation='sigmoid',kernel_initializer='random_uniform'))
# model.add(krs.layers.Dense(units=100, activation='sigmoid',kernel_initializer='random_uniform'))
# model.add(krs.layers.Dense(units=1, activation='relu',kernel_initializer='random_uniform'))
# model.add(krs.layers.Dense(units=128, activation='sigmoid'))
# model.add(krs.layers.Dense(units=64, activation='relu'))
# model.add(krs.layers.Dense(units=32, activation='sigmoid'))
# model.add(krs.layers.Dense(units=1, activation='relu'))
model.add(krs.layers.Dense(units= 12, activation='sigmoid'))
model.add(krs.layers.Dense(units= 12))
model.add(krs.layers.advanced_activations.LeakyReLU(alpha=0.3))
model.add(krs.layers.Dense(units=1, activation='relu'))


# sgd = krs.optimizers.SGD(lr=0.1)
# sgd = krs.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer='sgd', loss='mean_absolute_error')


krs.utils.plot_model(model, to_file='model.png', show_shapes=True)

model.fit(modulationSamplesTrain, trueDepthsTrain, epochs=epochs, batch_size=batch_size)
# model.fit(modulationSamplesTrain.reshape(numTrainSamples,1,numPoints,k), trueDepthsTrain.reshape(numTrainSamples,1,1,1), epochs=epochs, batch_size=batch_size)

loss_and_metrics = model.evaluate(modulationSamplesTest, trueDepthsTest, batch_size=batch_size)
# loss_and_metrics = model.evaluate(modulationSamplesTest.reshape(numTestSamples,1,numPoints,k), trueDepthsTest.reshape(numTestSamples,1,1,1), batch_size=batch_size)

# print ''
# print loss_and_metrics

# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
# 	json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")
 


