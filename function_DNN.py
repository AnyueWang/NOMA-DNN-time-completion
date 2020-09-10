from __future__ import print_function
import tensorflow as tf
import numpy as np
import math
import scipy.io as sio
import time

#----------------------------------------------------------------------------------------------------------------------
# Functions for deep neural network weights initialization
def ini_weights(n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4,
                n_hidden_5, n_hidden_6, n_hidden_7, n_hidden_8,
                n_hidden_9, n_hidden_10, n_output):
    weights = {

        'hidden_1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],stddev=1/math.sqrt(n_input))), #variable: ; truncated_norm: stochastic no more than 2 standard variance
        'hidden_2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=1 / math.sqrt(n_hidden_1))),
        'hidden_3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=1 / math.sqrt(n_hidden_2))),
        'hidden_4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4], stddev=1 / math.sqrt(n_hidden_3))),
        'hidden_5': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_5], stddev=1 / math.sqrt(n_hidden_4))),
        'hidden_6': tf.Variable(tf.truncated_normal([n_hidden_5, n_hidden_6], stddev=1 / math.sqrt(n_hidden_5))),
        'hidden_7': tf.Variable(tf.truncated_normal([n_hidden_6, n_hidden_7], stddev=1 / math.sqrt(n_hidden_6))),
        'hidden_8': tf.Variable(tf.truncated_normal([n_hidden_7, n_hidden_8], stddev=1 / math.sqrt(n_hidden_7))),
        'hidden_9': tf.Variable(tf.truncated_normal([n_hidden_8, n_hidden_9], stddev=1 / math.sqrt(n_hidden_8))),
        'hidden_10': tf.Variable(tf.truncated_normal([n_hidden_9, n_hidden_10], stddev=1 / math.sqrt(n_hidden_9))),
        'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_output],stddev=1/ math.sqrt(n_hidden_4))),
    }
    tf.add_to_collection(
        'losses',tf.contrib.layers.l2_regularizer(0.0003)(weights['hidden_1']))
    tf.add_to_collection(
        'losses',tf.contrib.layers.l2_regularizer(0.0003)(weights['hidden_2']))
    tf.add_to_collection(
        'losses',tf.contrib.layers.l2_regularizer(0.0003)(weights['hidden_3']))
    tf.add_to_collection(
        'losses',tf.contrib.layers.l2_regularizer(0.0003)(weights['hidden_4']))
    tf.add_to_collection(
        'losses', tf.contrib.layers.l2_regularizer(0.0003)(weights['hidden_5']))
    tf.add_to_collection(
        'losses', tf.contrib.layers.l2_regularizer(0.0003)(weights['hidden_6']))
    tf.add_to_collection(
        'losses', tf.contrib.layers.l2_regularizer(0.0003)(weights['hidden_7']))
    tf.add_to_collection(
        'losses', tf.contrib.layers.l2_regularizer(0.0003)(weights['hidden_8']))
    tf.add_to_collection(
        'losses', tf.contrib.layers.l2_regularizer(0.0003)(weights['hidden_9']))
    tf.add_to_collection(
        'losses', tf.contrib.layers.l2_regularizer(0.0003)(weights['hidden_10']))
    biases = {
        'b_hidden_1': tf.Variable(tf.zeros([n_hidden_1])),
        'b_hidden_2': tf.Variable(tf.zeros([n_hidden_2])),
        'b_hidden_3': tf.Variable(tf.zeros([n_hidden_3])),
        'b_hidden_4': tf.Variable(tf.zeros([n_hidden_4])),
        'b_hidden_5': tf.Variable(tf.zeros([n_hidden_5])),
        'b_hidden_6': tf.Variable(tf.zeros([n_hidden_6])),
        'b_hidden_7': tf.Variable(tf.zeros([n_hidden_7])),
        'b_hidden_8': tf.Variable(tf.zeros([n_hidden_8])),
        'b_hidden_9': tf.Variable(tf.zeros([n_hidden_9])),
        'b_hidden_10': tf.Variable(tf.zeros([n_hidden_10])),
        'out': tf.Variable(tf.zeros([n_output])),
    }
    return weights, biases
#----------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------

def forward_propagation(x, weights, biases):
    layer_1 = tf.nn.relu(tf.matmul(x, weights['hidden_1']) + biases['b_hidden_1'])
    layer_2 = tf.nn.relu(tf.matmul(layer_1, weights['hidden_2']) + biases['b_hidden_2'])
    layer_3 = tf.nn.relu(tf.matmul(layer_2, weights['hidden_3']) + biases['b_hidden_3'])
    layer_4 = tf.nn.relu(tf.matmul(layer_3, weights['hidden_4']) + biases['b_hidden_4'])
    layer_5 = tf.nn.relu(tf.matmul(layer_4, weights['hidden_5']) + biases['b_hidden_5'])
    layer_6 = tf.nn.relu(tf.matmul(layer_5, weights['hidden_6']) + biases['b_hidden_6'])
    layer_7 = tf.nn.relu(tf.matmul(layer_6, weights['hidden_7']) + biases['b_hidden_7'])
    layer_8 = tf.nn.relu(tf.matmul(layer_7, weights['hidden_8']) + biases['b_hidden_8'])
    layer_9 = tf.nn.relu(tf.matmul(layer_8, weights['hidden_9']) + biases['b_hidden_9'])
    layer_10 = tf.nn.relu(tf.matmul(layer_9, weights['hidden_10']) + biases['b_hidden_10'])
    out_layer = tf.nn.softmax(tf.matmul(layer_4, weights['out']) + biases['out'])
    #out_layer = 2 * tf.nn.sigmoid(tf.matmul(layer_4, weights['out']) + biases['out'])+1
    #out_layer = tf.matmul(layer_10, weights['out']) + biases['out']

    return out_layer
#----------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------
# Functions for deep neural network training
def train_DNN(X, Y, X_t, Y_t, location, training_epochs, batch_size, LR,
              n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4,
              n_hidden_5, n_hidden_6, n_hidden_7, n_hidden_8,
              n_hidden_9, n_hidden_10,
              traintestsplit, LRdecay):
    num_total = X.shape[1]  # number of total samples

    #Y = np.reshape(Y, (2 * K, num_total))  # reshape data to column

    num_val = int(num_total * traintestsplit)  # number of validation samples
    num_train = num_total - num_val  # number of training samples
    n_input = X.shape[0]  # input size
    n_output = Y.shape[0]  # output size
    X_train = np.transpose(X[:, 0:num_train])  # training data
    Y_train = np.transpose(Y[:, 0:num_train])  # training label
    #X_val = np.transpose(X[:, num_train:num_total])  # validation data
    #Y_val = np.transpose(Y[:, num_train:num_total])  # validation label

    tf.reset_default_graph()
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_output])
    is_train = tf.placeholder("bool")
    learning_rate = tf.placeholder(tf.float32, shape=[])
    total_batch = int(num_total / batch_size)
    print('train: %d ' % num_train, 'validation: %d ' % num_val)


    weights, biases = ini_weights(n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4,
                                  n_hidden_5, n_hidden_6, n_hidden_7, n_hidden_8,
                                  n_hidden_9, n_hidden_10, n_output)
    pred = forward_propagation(x, weights, biases)

    cost = tf.reduce_mean(tf.square(pred - y))  # cost function: MSE
    #cost = tf.reduce_mean(-y * tf.log(pred) + (1 - y) * tf.log(1 - pred))  # loss fucntion: cross entropy
    #cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y, logits=pred))
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
    tf.add_to_collection('losses', cost)
    cost = tf.add_n(tf.get_collection('losses'))
    tf.summary.scalar("loss", cost)
    #optimizer2 = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)  # training algorithms
    optimizer2 = tf.train.AdamOptimizer(learning_rate).minimize(cost)  # training algorithms

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    Metric = np.zeros((training_epochs, 3))
    with tf.Session() as sess:
        sess.run(init)
        acc = np.zeros(training_epochs)
        for epoch in range(training_epochs):
            for i in range(total_batch):
                idx = np.random.randint(num_train, size=batch_size)
                if LRdecay == 1:
                    _, c = sess.run([optimizer2, cost], feed_dict={x: X_train[idx, :], y: Y_train[idx, :],
                                                                     learning_rate: LR / (epoch + 1), is_train: True})
                elif LRdecay == 0:
                    _, c = sess.run([optimizer2, cost], feed_dict={x: X_train[idx, :], y: Y_train[idx, :],
                                                                     learning_rate: LR, is_train: True})
            # loss value in training
            print("train_loss_epoch_%i" % epoch, "=%f" % c)
            Metric[epoch, 0] = c
            # loss value in evaluating
            pred_test, c_test = sess.run([pred, cost], feed_dict={x: np.transpose(X_t), y: np.transpose(Y_t), is_train: False})
            print("test_loss_epoch_%i" % epoch, "=%f" % c_test)
            Metric[epoch, 1] = c_test
            pred_mean = np.mean(pred_test, axis=1)
            #print(pred_mean)
            y_round = np.round(pred_test)
            #print('round=',np.shape(Y_t))
            num_test = 500
            false_num = 0
            for n in range(num_test):
                match_pos = (y_round[n, :] - Y_t[:, n]==0)
                #print(match_pos)
                false_size = np.shape(np.where(match_pos == False))
                if false_size[1] > 0:
                    false_num = false_num + 1
            #print('match_num=', match_size[1])

            acc[epoch] = 1 - false_num/500
            #acc[epoch] = match_pos / 500 / 10
            print("test_acc_epoch_%i" % epoch, "=%f" % acc[epoch])
            Metric[epoch, 2] = acc[epoch]

            #print(biases['b_hidden_1'].eval())

        saver.save(sess, location)
    return Metric


# Functions for deep neural network testing
def test_DNN(X, model_location,
             n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4,
             n_hidden_5, n_hidden_6, n_hidden_7, n_hidden_8,
             n_hidden_9, n_hidden_10, n_output):
    n_input = X.shape[0]

    tf.reset_default_graph()
    x = tf.placeholder("float", [None, n_input])
    is_train = tf.placeholder("bool")
    weights, biases = ini_weights(n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4,
                                  n_hidden_5, n_hidden_6, n_hidden_7, n_hidden_8,
                                  n_hidden_9, n_hidden_10, n_output)
    pred = forward_propagation(x, weights, biases)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_location)
        y_pred = sess.run(pred, feed_dict={x: np.transpose(X), is_train: False})

    return y_pred
#----------------------------------------------------------------------------------------------------------------------