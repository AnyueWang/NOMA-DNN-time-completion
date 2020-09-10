import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gurobipy
import scipy.io as sio                     # import scipy.io for .mat file I/O
import numpy as np                         # import numpy
import matplotlib.pyplot as plt            # import matplotlib.pyplot for figure plotting
import itertools
import time
import sys
#-----------------------------------------------------------------------------------------------------------------------
import function_DNN as df     # 2-layer full-connection neural network
#-----------------------------------------------------------------------------------------------------------------------
num_H = 3500              # number of training samples
batch_size = 16
training_epochs = 1000      # number of training epochs
LR = 0.0001 # learning rate
# #-----------------------------------------------------------------------------------------------------------------------

X_file = ''
X_dict = sio.loadmat(X_file) # import scipy.io for .mat file I/O
#X = np.transpose(X_dict['H_sample'])
X = np.transpose(X_dict['HD_sample'])
#X = X_dict['H_sample']
X_train = X[0:num_H, :]
X_test = X[5000:6000, :]
X_train = np.transpose(X_train)
X_test = np.transpose(X_test)
sio.savemat('X_train', {'X_train': X_train})
sio.savemat('X_test', {'X_test': X_test})

Y_file = ''
Y_dict = sio.loadmat(Y_file)
Y = np.transpose(Y_dict['y_new'])
Y_train = Y[0:num_H, :]
Y_test = Y[5000:6000, :]
Y_train = np.transpose(Y_train)
Y_test = np.transpose(Y_test)
sio.savemat('Y_train', {'Y_train': Y_train})
sio.savemat('Y_test', {'Y_test': Y_test})


# Training and Test Deep Neural Networks
#---------------------------------------------------DNN-------------------------------------------------------------
n_hidden_1 = 32
n_hidden_2 = 32
n_hidden_3 = 32
n_hidden_4 = 32
n_hidden_5 = 32
n_hidden_6 = 32
n_hidden_7 = 32
n_hidden_8 = 32
n_hidden_9 = 32
n_hidden_10 = 32

model_location_DNN = "./DNNmodel/model_DNN_MFN.ckpt" # ？

start_1 = time.clock()
Metric = df.train_DNN(X_train, Y_train, X_test, Y_test,
                      model_location_DNN, training_epochs, batch_size, LR,
                      n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4,
                      n_hidden_5, n_hidden_6, n_hidden_7, n_hidden_8,
                      n_hidden_9, n_hidden_10,
                      0, 0)
start_2 = time.clock()
train_time = start_2 - start_1
print('ompl_train_time = %f' % train_time)
sio.savemat('Metric', {'Metric': Metric})
start_3 = time.clock()
Y_pridict = df.test_DNN(X_test, model_location_DNN,
                        n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4,
                        n_hidden_5, n_hidden_6, n_hidden_7, n_hidden_8,
                        n_hidden_9, n_hidden_10,
                        Y_test.shape[0])
start_4 = time.clock()
train_time = start_4 - start_3
print('ompl_train_time = %f' % train_time)
sio.savemat('Y_pridict', {'Y_pridict': Y_pridict})
#-------------------------------------------------------------------------------------------------------------------
