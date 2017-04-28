import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Lambda
# from customKerasActivation import MyLayer
# model = Sequential([
#     Dense(32, input_dim=784),
#     Activation('relu'),
#     Dense(10),
#     Activation('softmax'),
# ])



# For a single-input model with 10 classes (categorical classification):

def customF(x):
	print 'customF'
	print x
	return x + 1

def customOutShape(shape):
	return shape
# def customF_output_shape(x):
# 	return x.shape

model = Sequential()
model.add(Dense(1, activation='sigmoid', input_dim=100))
customL = Lambda(customF,output_shape=customOutShape)
model.add(customL)
# model.add(Dense(32, activation='sigmoid', input_dim=100))
# model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((20, 5))
labels = np.random.randint(5, size=(20, 1))

# Convert labels to categorical one-hot encoding
binary_labels = keras.utils.np_utils.to_categorical(labels, 10)

# Train the model, iterating on the data in batches of 32 samples
# model.fit(data, binary_labels,  batch_size=32, epochs=10)
model.fit(data, binary_labels,  batch_size=1, nb_epoch=20)

