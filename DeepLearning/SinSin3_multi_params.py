# Usage:
# $ source activate mlenv
# uncomment the params you want to run
# change the folder path

# 2 - 3: different activation
# 4 - 6: different learning rate
# 7:     different initlizers
# 8 - 9: different batch size
# 10 - 16: different layers

########################################################################################################

params = [
    {'activation': ['leakyrelu', 0.3], 'rate': 0.01, 'dynamic_rate': False, 'batch': 128, 'layers': [12, 12],       'init': 'random_uniform'},   # 1
    
    # {'activation': ['leakyrelu', 0.1], 'rate': 0.01, 'dynamic_rate': False, 'batch': 128, 'layers': [12, 12],       'init': 'random_uniform'},   # 2
    # {'activation': ['linear'],         'rate': 0.01, 'dynamic_rate': False, 'batch': 128, 'layers': [12, 12],       'init': 'random_uniform'},   # 3
    
    # {'activation': ['leakyrelu', 0.3], 'rate': 0.05, 'dynamic_rate': False, 'batch': 128, 'layers': [12, 12],       'init': 'random_uniform'},   # 4
    # {'activation': ['leakyrelu', 0.3], 'rate': 0.1,  'dynamic_rate': False, 'batch': 128, 'layers': [12, 12],       'init': 'random_uniform'},   # 5
    # {'activation': ['leakyrelu', 0.3], 'rate': 0.1,  'dynamic_rate': True,  'batch': 128, 'layers': [12, 12],       'init': 'random_uniform'},   # 6

    # {'activation': ['leakyrelu', 0.3], 'rate': 0.01, 'dynamic_rate': False, 'batch': 128, 'layers': [12, 12],       'init': 'random_normal' },   # 7

    # {'activation': ['leakyrelu', 0.3], 'rate': 0.01, 'dynamic_rate': False, 'batch': 64,  'layers': [12, 12],       'init': 'random_uniform'},   # 8
    # {'activation': ['leakyrelu', 0.3], 'rate': 0.01, 'dynamic_rate': False, 'batch': 32,  'layers': [12, 12],       'init': 'random_uniform'},   # 9

    # {'activation': ['leakyrelu', 0.3], 'rate': 0.01, 'dynamic_rate': False, 'batch': 128, 'layers': [10],           'init': 'random_uniform'},   # 10
    # {'activation': ['leakyrelu', 0.3], 'rate': 0.01, 'dynamic_rate': False, 'batch': 128, 'layers': [50],           'init': 'random_uniform'},   # 11
    # {'activation': ['leakyrelu', 0.3], 'rate': 0.01, 'dynamic_rate': False, 'batch': 128, 'layers': [200],          'init': 'random_uniform'},   # 12
    # {'activation': ['leakyrelu', 0.3], 'rate': 0.01, 'dynamic_rate': False, 'batch': 128, 'layers': [100, 50],      'init': 'random_uniform'},   # 13
    # {'activation': ['leakyrelu', 0.3], 'rate': 0.01, 'dynamic_rate': False, 'batch': 128, 'layers': [400, 20],      'init': 'random_uniform'},   # 14
    # {'activation': ['leakyrelu', 0.3], 'rate': 0.01, 'dynamic_rate': False, 'batch': 128, 'layers': [40, 20, 10],   'init': 'random_uniform'},   # 15
    # {'activation': ['leakyrelu', 0.3], 'rate': 0.01, 'dynamic_rate': False, 'batch': 128, 'layers': [200, 100, 50], 'init': 'random_uniform'},   # 16
]

data_folder = '/Users/zhicheng/Desktop/SinusoidSinusoid3/'

########################################################################################################


from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from keras import backend as K
import random
from keras import layers, optimizers
from keras.callbacks import LearningRateScheduler, EarlyStopping

data_file_name = 'SinusoidSinusoid3_rand_rand.csv';


def read_data(file):
    data, label = [], []
    with open(data_folder + file) as f:
        for line in f:
            line = line.strip('\n')
            line_split = line.split(',')
            data.append([float(line_split[1]), float(line_split[2]), float(line_split[3])])
            label.append([float(line_split[0])])
    return np.array(data), np.array(label)


data, label = read_data(data_file_name)
test_data, test_label = data[:10000], label[:10000]
tune_data, tune_label = data[10000:20000], label[10000:20000]
train_data, train_label = data[20000:], label[20000:]


for param in params:
    model = Sequential()

    activation = layers.advanced_activations.LeakyReLU(alpha=param['activation'][1]) if param['activation'][0] == 'leakyrelu' else 'linear'

    model.add(Dense(units=param['layers'][0], 
                    input_dim=3, 
                    activation='sigmoid', 
                    kernel_initializer=param['init']))

    if len(param['layers']) > 1:
        if param['activation'][0] == 'leakyrelu':
            model.add(Dense(units=param['layers'][1], 
                            kernel_initializer=param['init']))
            model.add(layers.advanced_activations.LeakyReLU(alpha=param['activation'][1]))
        else:
            model.add(Dense(units=param['layers'][1], 
                            activation='linear',
                            kernel_initializer=param['init']))

    if len(param['layers']) > 2:
        if param['activation'][0] == 'leakyrelu':
            model.add(Dense(units=param['layers'][2], 
                            kernel_initializer=param['init']))
            model.add(layers.advanced_activations.LeakyReLU(alpha=param['activation'][1]))
        else:
            model.add(Dense(units=param['layers'][2], 
                            activation='linear',
                            kernel_initializer=param['init']))

    if param['activation'][0] == 'leakyrelu':
        model.add(Dense(units=1, 
                        kernel_initializer=param['init']))
        model.add(layers.advanced_activations.LeakyReLU(alpha=param['activation'][1]))
    else:
        model.add(Dense(units=1, 
                        activation='linear',
                        kernel_initializer=param['init']))


    sgd = optimizers.SGD(lr=param['rate'])
    model.compile(optimizer=sgd, loss='mean_absolute_error')

    # For dynamic learning rate.
    def scheduler(epoch):
        if epoch % 5 == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr*.9)
            print("lr changed to {}".format(lr*.9))
        return K.get_value(model.optimizer.lr)
    lr_decay = LearningRateScheduler(scheduler)
    

    earlyStopping=EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='auto')

    callbacks = [lr_decay, earlyStopping] if param['dynamic_rate'] else [earlyStopping]
    history_callback = model.fit(train_data, train_label, epochs=1000, batch_size=param['batch'], callbacks=callbacks, validation_split=0.0, validation_data=(tune_data, tune_label), shuffle=True)

    loss_history = history_callback.history["loss"]
    val_loss_history = history_callback.history["val_loss"]
    history = [(loss_history[i], val_loss_history[i]) for i in range(len(loss_history))]

    loss_and_metrics = model.evaluate(test_data, test_label, batch_size=param['batch'])

    print ''
    print ''
    print '====================================================================='
    print 'Result: '
    print 'Params: ', param
    print 'Loss: ', loss_and_metrics
    print '====================================================================='
    print ''

    # Save the results.
    output_name = ''
    for key, value in sorted(param.items()):
        output_name += str(key) + '_'
        if key in ['activation', 'layers']:
            output_name += '_'.join(str(e) for e in value) + '_'
        else:
            output_name += str(value) + '_'

    with open('results/' + output_name + '.txt', 'w') as f:
        f.write('Params: ' + str(param) + '\n')
        f.write('Loss: ' + str(loss_and_metrics) + '\n')
        f.write('\n')
        for (e1, e2) in history:
            f.write(str(e1) + ',' + str(e2) + '\n')

    # Save models.
    with open('results/' + output_name + '.json', "w") as json_file:
        json_file.write(model.to_json())
    model.save_weights('results/' + output_name + '.h5')

