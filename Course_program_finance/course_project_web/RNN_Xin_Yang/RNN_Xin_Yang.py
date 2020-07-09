# coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys

rnn_unit = 10  # number of hidden layers
input_size = 5
output_size = 1
lr = 0.0006  # learning rate
close_column = 1  # column number of close in stripped data
stock_code = sys.argv[1]


f = open('./data/' + stock_code + '_history.csv')
df = pd.read_csv(f)  # read csv data
data = df.iloc[:, 1:6].values  # choose column 2-6

#  ——————————————————get training set——————————————————
def get_train_data(batch_size=60, time_step=20, train_begin=0, train_end=800):
    batch_index = []
    data_train = data[train_begin:train_end]
    normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # regularization
    train_x, train_y = [], []
    for i in range(len(normalized_train_data) - time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i:i + time_step, :5]
        y = normalized_train_data[i + 1:i + time_step + 1, close_column, np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data) - time_step))
    return batch_index, train_x, train_y


# ——————————————————get test set——————————————————
def get_test_data(time_step=1, test_begin=1023):
    data_test = data[test_begin:]
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)
    normalized_test_data = (data_test - mean) / std  # regularization
    size = (len(normalized_test_data) + time_step) // time_step  # size of samples
    test_x, test_y = [], []
    for i in range(size):
        x = normalized_test_data[i * time_step:(i + 1) * time_step, :5]
        y = normalized_test_data[i * time_step + 1:(i + 1) * time_step + 1, close_column]
        test_x.append(x.tolist())
        test_y.extend(y)
    return mean, std, test_x, test_y


# ——————————————————define neural network structure——————————————————
weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}


# ——————————————————define neural network variables——————————————————
def lstm(X):
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, input_size])  # 2d tensor for computation, result as the input of hidden layer
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 3d tensor as input of lsmt cell
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = tf.reshape(output_rnn, [-1, rnn_unit])
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states


# ————————————————train model————————————————————

def train_lstm(batch_size=60, time_step=20, train_begin=200, train_end=800):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)
    with tf.variable_scope("sec_lstm"):
        pred, _ = lstm(X)
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(200):  # iteration times
            for step in range(len(batch_index) - 1):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                                 Y: train_y[batch_index[step]:batch_index[step + 1]]})
                # print("Number of iterations:", i, " loss:", loss_)
        saver.save(sess, 'model_save/model.ckpt')
        # print("model_save: ", saver.save(sess, 'model_save/model.ckpt'))
        # print("The training has finished")


# ————————————————prediction————————————————————
def prediction(time_step=20, test_begin=800):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    mean, std, test_x, test_y = get_test_data(time_step, test_begin)
    with tf.variable_scope("sec_lstm", reuse=True):
        pred, _ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # restore parameters
        module_file = tf.train.latest_checkpoint('model_save')
        saver.restore(sess, module_file)
        test_predict = []
        for step in range(len(test_x) - 1):
            prob = sess.run(pred, feed_dict={X: [test_x[step]]})
            predict = prob.reshape((-1))
            test_predict.extend(predict)
        test_y = np.array(test_y) * std[close_column] + mean[close_column]
        test_predict = np.array(test_predict) * std[close_column] + mean[close_column]
        # print(test_predict)
        predicted_price = test_predict[len(test_predict) - 1]
        '''
        acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  # accuracy
        print("The accuracy of this predict:",acc)
        '''

        # plot line chart
        '''
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b', )
        plt.plot(list(range(len(test_y))), test_y, color='r')
        plt.show()
        '''
        sess.close()
        return predicted_price


if __name__ == '__main__':
    train_lstm(60, 20, 0, len(data) - 20)
    predicted_price = prediction(1, len(data) - 20)
    # print(len(data))
    print(predicted_price)
    sys.exit(0)