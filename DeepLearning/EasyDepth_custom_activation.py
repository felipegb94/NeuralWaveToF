# This one is not working currently.

from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from keras import backend as K


# def spiky(x):
#     r = x % 1
#     if r <= 0.5:
#         return r
#     else:
#         return 0

# np_spiky = np.vectorize(spiky)

# def d_spiky(x):
#     r = x % 1
#     if r <= 0.5:
#         return 1
#     else:
#         return 0
# np_d_spiky = np.vectorize(d_spiky)

# np_d_spiky_32 = lambda x: np_d_spiky(x).astype(np.float32)

# def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

#     # Need to generate a unique name to avoid duplicates:
#     rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

#     tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
#     g = tf.get_default_graph()
#     with g.gradient_override_map({"PyFunc": rnd_name}):
#         return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

# def tf_d_spiky(x,name=None):
#     with ops.op_scope([x], name, "d_spiky") as name:
#         y = tf.py_func(np_d_spiky_32,
#                         [x],
#                         [tf.float32],
#                         name=name,
#                         stateful=False)
#         return y[0]

# def spikygrad(op, grad):
#     x = op.inputs[0]

#     n_gr = tf_d_spiky(x)
#     return grad * n_gr  

# np_spiky_32 = lambda x: np_spiky(x).astype(np.float32)

# def tf_spiky(x, name=None):

#     with ops.op_scope([x], name, "spiky") as name:
#         y = py_func(np_spiky_32,
#                         [x],
#                         [tf.float32],
#                         name=name,
#                         grad=spikygrad)  # <-- here's the call to the gradient
#         return y[0]


def custom_activation(x):
    # return K.sigmoid(x + K.random_normal_variable((K.ndim(x), 1), 0, x))
    # noise = K.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
    noise = K.random_normal(shape=K.shape(x), mean=0.0, stddev=x, dtype=K.dtype(x)) 
    return K.sigmoid(x + noise)
    # return K.sigmoid(x)


model = Sequential()

model.add(Dense(units=32, input_dim=3, activation=custom_activation))
model.add(Dense(units=1, activation='relu'))

model.compile(optimizer='adam', loss='mean_squared_error')



datas = np.random.random((100, 3)) * 10
labels = []
for data in datas:
    labels.append([3 * data[0] + 2 * data[1] - data[2]])
labels = np.array(labels)



model.fit(datas[:80], labels[:80], epochs=500, batch_size=10)

print labels[80:]
print model.predict(datas[80:], batch_size=10)