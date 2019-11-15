import numpy as np
import tensorflow as tf
from timeit import default_timer as timer
import pickle

from read_data import read_data_chars
from models import char_different_rnn

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

MAX_DOCUMENT_LENGTH = 100
MAX_LABEL = 15
epochs = 1000
lr = 0.01
batch_size = 128

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)


x_train, y_train, x_test, y_test = read_data_chars()

x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
y_ = tf.placeholder(tf.int64)
keep_prob = tf.placeholder(tf.float32)

states, logits = char_different_rnn(x, 'rnn')

# Optimizer
entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

# Testing
correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(tf.one_hot(y_, MAX_LABEL), 1)), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

N = len(x_train)
idx = np.arange(N)

# training
training_loss = []
testing_acc = []
start = timer()
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for e in range(epochs):
    np.random.shuffle(idx)
    x_train = x_train[idx]
    y_train = y_train[idx]
    for start in range(0, N - batch_size, batch_size):
      train_op.run(
        feed_dict={x: x_train[start: start + batch_size], y_: y_train[start: start + batch_size], keep_prob: 1})

    loss_ = entropy.eval(feed_dict={x: x_train, y_: y_train, keep_prob: 1})
    acc_ = accuracy.eval(feed_dict={x: x_test, y_: y_test, keep_prob: 1})
    training_loss.append(loss_)
    testing_acc.append(acc_)

    if e % 1 == 0:
      print('\riter: %d, entropy: %g, testing accuracy: %g' % (e, training_loss[e], testing_acc[e]), end='')


end = timer()
print(end - start) # Time in seconds, e.g. 5.38091952400282
with open('../output/char_rnn_rnn.pkl', 'wb') as f:
    stat = {'train_loss': training_loss, 'test_acc': testing_acc}
    pickle.dump(stat, f)

