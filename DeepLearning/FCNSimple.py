# from __future__ import print_function

import numpy as np

import DataUtils

import keras
from keras import utils
from keras import layers
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.convolutional import (Conv2D, Conv2DTranspose, MaxPooling2D, ZeroPadding2D, Cropping2D)

	# return (bAll)


(trueDepthsTrain, bTrain, trueDepthsTune, bTune, trueDepthsTest, bTest) = DataUtils.loadDataset()

#################### dataset parameters ##########################
(numTrainSamples, numRows, numCols,k) = bTrain.shape
(numTuneSamples) = bTune.shape[0]
(numTestSamples) = bTest.shape[0]
print "Train Samples: " + str(numTrainSamples)
print "Tune Samples: " + str(numTuneSamples)
print "Test Samples: " + str(numTestSamples)
print "Rows: " + str(numRows)
print "Cols: " + str(numCols)
print "k: " + str(k)

##################### training parameters ##########################
batchSize = 1
epochs = 100
learningRate = 0.01
##################### model definition ##########################

model = Sequential()
######### Convolution layers
model.add(Conv2D(filters=64, kernel_size=(3,3),padding="same",data_format="channels_last", use_bias= True,activation="relu",input_shape=(numRows,numCols,k)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1'))
model.add(Conv2D(filters=128, kernel_size=(3,3),padding="same",data_format="channels_last", use_bias= True,activation="relu",input_shape=(numRows,numCols,k)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2'))
# model.add(Conv2D(filters=256, kernel_size=(3,3),padding="same",data_format="channels_last", use_bias= True,activation="relu",input_shape=(numRows,numCols,k)))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3'))



######### Deconvolution layers
# model.add(Conv2DTranspose(filters=256, kernel_size=(3,3), 
# 	strides=(2, 2), 
# 	padding='same', 
# 	data_format="channels_last", activation="relu", use_bias=True, name="deconv1"))

model.add(Conv2DTranspose(filters=128, kernel_size=(3,3), 
	strides=(2, 2), 
	padding='same', 
	data_format="channels_last", activation="relu", use_bias=True, name="deconv2"))

model.add(Conv2DTranspose(filters=64, kernel_size=(3,3), 
	strides=(2, 2), 
	padding='same', 
	data_format="channels_last", activation="relu", use_bias=True, name="deconv3"))


model.add(Conv2D(filters=1, kernel_size=(3,3),padding="same",data_format="channels_last",name="output") )
model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.3))
# model.add(Conv2D(filters=1, kernel_size=(3,3),padding="same",data_format="channels_last", activation="relu") )


# keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
sgd = optimizers.SGD(lr=learningRate)
model.compile(optimizer=adam, loss='mean_absolute_error')

# utils.plot_model(model, to_file='FCNSimple.png', show_shapes=True)

earlyStopping=EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')


historyCallback = model.fit(bTrain,trueDepthsTrain, batch_size=batchSize,epochs=epochs,callbacks=[earlyStopping],validation_data=(bTune,trueDepthsTune),shuffle=True)
valLossHistory = historyCallback.history["val_loss"]

lossAndMetrics = model.evaluate(bTest, trueDepthsTest)
# print lossAndMetrics

print ''
print ''
print '====================================================================='
print 'Result: '
print 'Loss: ', lossAndMetrics
print '====================================================================='
print ''

model_json = model.to_json()
with open("FCNSimple.json", "w") as json_file:
	json_file.write(model_json)
model.save_weights("FCNSimple" + '.h5')
print("Saved model to disk")

# estimatedDepthMap = model.predict(bTest)
# print estimatedDepthMap.shape
# np.savetxt("../estimatedDepthMap.csv", estimatedDepthMap[0], fmt="%.8f", delimiter=",")
# np.savetxt("../trueDepthMap.csv", trueDepthsTest[0], fmt="%.8f", delimiter=",")
# np.savetxt("../tuneDepthMap.csv", trueDepthsTune[0], fmt="%.8f", delimiter=",")
# np.savetxt("../trainDepthMap.csv", trueDepthsTrain[0], fmt="%.8f", delimiter=",")
