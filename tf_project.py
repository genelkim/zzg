

# Mean Squared Error Loss Function for TensorFlow.
def mse_loss_fn(prediction, gold):
  return tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(prediction, gold), 2)))



import numpy as np
np.random.seed(1335) # for reproducibility
np.set_printoptions(precision=5, suppress=True, linewidth=150)

from matplotlib import pyplot as plt


import quandl



#
# LSTM Definition
#

BATCH_SIZE = 1
NUM_FEATURES = 7

STEPSIZE = 1 

LSTM_HIDDEN_STATE = 64

# TODO: fill
W = tf.Variable(...) # weights
b = tf.Variable(...) # bias

lstm = tf.contrib.rnn.BasicLSTMCell(LSTM_HIDDEN_STATE)
# TODO: add dropout 0.5

initial_state = tf.zeros([BATCH_SIZE, lstm.state_size])
state = initial_state


for batch in batches: # TODO: define this...
  output, state = lstm(batch, state)

  prediction = tf.nn.softmax(tf.matmul(output, W) + b)

  loss = mse_loss_fn(prediction, gold)

  tf.train.AdamOptimizer(STEPSIZE).minimize(loss)



#
# Q-Learning (Main).
# 


# Q-Learning Parameters
gamma = 0.95  # discount factor
alpha = 1     # learning rate

# Global Parameters
EPOCHS = 10

sample_size = 100
buffer_size = 200

replay_buffer = []
reward_history = []

selected_actions = []








