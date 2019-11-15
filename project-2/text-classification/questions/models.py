import tensorflow as tf

MAX_DOCUMENT_LENGTH = 100
N_FILTERS = 10
FILTER_SHAPE1_CHAR = [20, 256]
FILTER_SHAPE1_WORD= [20, 20]
FILTER_SHAPE2 = [20, 1]
POOLING_WINDOW = 4
POOLING_STRIDE = 2
MAX_LABEL = 15
EMBEDDING_SIZE = 20
HIDDEN_SIZE = 20

epochs = 100
lr = 0.01
batch_size = 128

def char_cnn_model(x, keep_prob):

    input_layer = tf.reshape(
        tf.one_hot(x, 256), [-1, MAX_DOCUMENT_LENGTH, 256, 1])

    with tf.variable_scope('CNN_Layer1'):
        conv1 = tf.layers.conv2d(
            input_layer,
            filters=N_FILTERS,
            kernel_size=FILTER_SHAPE1_CHAR,
            padding='VALID',
            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(
            conv1,
            pool_size=POOLING_WINDOW,
            strides=POOLING_STRIDE,
            padding='SAME')
        pool1_dropout = tf.nn.dropout(pool1, keep_prob)

    with tf.variable_scope('CNN_Layer2'):
        conv2 = tf.layers.conv2d(
            pool1_dropout,
            filters=N_FILTERS,
            kernel_size=FILTER_SHAPE2,
            padding='VALID',
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(
            conv2,
            pool_size=POOLING_WINDOW,
            strides=POOLING_STRIDE,
            padding='SAME')
        pool2_dropout = tf.nn.dropout(pool2, keep_prob)

    pool2_squeeze = tf.squeeze(tf.reduce_max(pool2_dropout, 1), squeeze_dims=[1])

    logits = tf.layers.dense(pool2_squeeze, MAX_LABEL, activation=None)

    return logits


def word_cnn_model(x, n_words, keep_prob):
  with tf.variable_scope('CNN_WORD'):
    word_vectors = tf.contrib.layers.embed_sequence(
        x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

    input_layer = tf.reshape(
        word_vectors, [-1, MAX_DOCUMENT_LENGTH, EMBEDDING_SIZE, 1])

    conv1 = tf.layers.conv2d(
        input_layer,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE1_WORD,
        padding='VALID',
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME')
    pool1_dropout = tf.nn.dropout(pool1, keep_prob)

    conv2 = tf.layers.conv2d(
        pool1_dropout,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE2,
        padding='VALID',
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(
        conv2,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME')
    pool2_dropout = tf.nn.dropout(pool2, keep_prob)

    pool2_squeeze = tf.squeeze(tf.reduce_max(pool2_dropout, 1), squeeze_dims=[1])

    logits = tf.layers.dense(pool2_squeeze, MAX_LABEL, activation=None)

    return logits


def char_rnn_model(x, keep_prob):
  with tf.variable_scope('RNN_CHAR'):
    char_vectors = tf.one_hot(x, 256)

    char_list = tf.unstack(char_vectors, axis=1)
    cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE, reuse=tf.get_variable_scope().reuse)
    cell=tf.nn.rnn_cell.DropoutWrapper(
      cell,
      input_keep_prob=1,
      output_keep_prob=keep_prob,
      state_keep_prob=1,
    )
    _, encoding = tf.nn.static_rnn(cell, char_list, dtype=tf.float32)

    logits = tf.layers.dense(encoding, MAX_LABEL, activation=None, reuse=tf.get_variable_scope().reuse)

  return logits


def word_rnn_model(x, n_words, keep_prob):
  with tf.variable_scope('RNN_WORD'):

    word_vectors = tf.contrib.layers.embed_sequence(
        x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

    word_list = tf.unstack(word_vectors, axis=1)
    cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
    cell=tf.nn.rnn_cell.DropoutWrapper(
      cell,
      input_keep_prob=1,
      output_keep_prob=keep_prob,
      state_keep_prob=1,
    )
    _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

    logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

  return logits


def char_different_rnn(x, cell_type):

    with tf.variable_scope('RNN_CHAR_LSTM'):
        char_vectors = tf.one_hot(x, 256)
        char_list = tf.unstack(char_vectors, axis=1)

        if cell_type == "vanilla_rnn":
            cell = tf.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE)
        elif cell_type == "lstm":
            cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        elif cell_type == "double_gru":
            cell1 = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
            cell2 = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

        outputs, state = tf.nn.static_rnn(cell, char_list, dtype=tf.float32)
        if cell_type == "double_gru" or cell_type == "lstm":
            state = state[-1]
        logits = tf.layers.dense(state, MAX_LABEL, activation=None)

    return state, logits


def word_different_rnn(x, n_words, cell_type):
    with tf.variable_scope('qns_6_WORD'):

        word_vectors = tf.contrib.layers.embed_sequence(
            x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

        word_list = tf.unstack(word_vectors, axis=1)

        if cell_type == "vanilla_rnn":
            cell = tf.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE)
        elif cell_type == "lstm":
            cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        elif cell_type == "double_gru":
            cell1 = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
            cell2 = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

        outputs, state = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)
        if cell_type == "double_gru" or cell_type == "lstm":
            state = state[-1]
        logits = tf.layers.dense(state, MAX_LABEL, activation=None)

    return logits