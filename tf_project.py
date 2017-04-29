
import numpy as np
np.random.seed(1335) # for reproducibility
#np.set_printoptions(precision=5, suppress=True, linewidth=150)

from matplotlib import pyplot as plt
import tensorflow as tf
import numpy.random as random













# Data format:
# Open, Close, Low, High, Volume
data = np.genfromtxt('data/sorted_data.csv.10000', delimiter=",")[1:,2:]

# Data subset
data = data[:10000]


#
# Global Parameters.
#

BUY_ACT = 2
SELL_ACT = 0
HOLD_ACT = 1
ACTION_LIST = [SELL_ACT, HOLD_ACT, BUY_ACT]
NUM_ACTIONS = len(ACTION_LIST)
ACTION_VAL = { BUY_ACT : 100.0, SELL_ACT : -100.0, HOLD_ACT : 0.0 }


# Mean Squared Error Loss Function for TensorFlow.
def mse_loss_fn(prediction, gold):
  return tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(tf.transpose(prediction), gold), 2)))

# Index of data for price(close)
PRICE_INDEX = 1

# Reward is difference in price(close) difference * \
#           action (number of shares sold/bought)
def get_reward(action, t, data):
  return (data[t][PRICE_INDEX] - data[t-1][PRICE_INDEX])*ACTION_VAL[action]


# Evaluate performance on optimal strategy (no epsilon or random action).
def max_reward(sess, data):
  lstate = np.random.normal(0, 0.2, 2*LSTM_HIDDEN_SIZE).astype(np.float32).reshape((1,10))
  state = data[0]
  reward_sum = 0

  rewards = []
  actions = []
  for t in xrange(1, len(data)):
    qvals_eval, new_lstm_state_eval = \
        sess.run([qvals, new_lstm_state], feed_dict={inputs:state.reshape((1,5)), lstm_state:lstate})
    action = np.argmax(qvals_eval)
    actions.append(action)

    reward = get_reward(action, t, data)
    reward_sum += reward
    #rewards.append(reward)

    lstate = new_lstm_state_eval
    state = data[t:t+1]
    
  print np.bincount(actions)
  return reward_sum


#
# LSTM Definition
#

BATCH_SIZE = 1
NUM_FEATURES = 5

STEPSIZE = 1 

#LSTM_HIDDEN_SIZE = 64
LSTM_HIDDEN_SIZE = NUM_FEATURES

#
# Tensorflow LSTM with Fully Conncted output layer.
#

# Placeholders that are supplied each call:
#   input for current step
#   lstm hidden state from previous step
#   gold value for training
inputs = tf.placeholder(tf.float32, shape=(BATCH_SIZE,NUM_FEATURES), name="tf_inputs")
gold = tf.placeholder(tf.float32, shape=(BATCH_SIZE,NUM_ACTIONS), name="tf_gold")

# c - LSTM speak
lstm_state = tf.placeholder(tf.float32, shape=(BATCH_SIZE,2*LSTM_HIDDEN_SIZE), name="tf_lstm_state")

# Fully Connected layer weights and biases.
FW = tf.Variable(tf.truncated_normal([NUM_ACTIONS, LSTM_HIDDEN_SIZE], stddev=0.2, mean=0, dtype=tf.float32), name="FW", dtype=tf.float32)
FB = tf.Variable(tf.truncated_normal([NUM_ACTIONS, 1], stddev=0.2, mean=0, dtype=tf.float32), name="FB", dtype=tf.float32)

# LSTM model (weights are stored implicitly)
lstm = tf.contrib.rnn.BasicLSTMCell(LSTM_HIDDEN_SIZE, state_is_tuple=False)
# TODO: add dropout 0.5

# Prediction.
#lstm_in = tf.concat([inputs, lstm_state], 0)
outputs, new_lstm_state = lstm(inputs, lstm_state)
#outputs = pred[0]
#new_lstm_state = pred[1]
qvals = tf.matmul(FW, tf.transpose(outputs))

# Training.
loss = mse_loss_fn(qvals, gold)
train_step = tf.train.AdamOptimizer(STEPSIZE).minimize(loss)

# Start interactive session.
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


#
# Q-Learning (Main).
# 


# Q-Learning Parameters
gamma = 0.01  # discount factor
alpha = 1     # learning rate
epsilon = 1   # exploration factor

# Global Parameters
EPOCHS = 10

BUFFER_SIZE = 200

replay_buffer = []
action_history = []

WINDOW_SIZE = 15


# Train.
for k in xrange(EPOCHS):

  # Initialize state and data.
  lstate = np.random.normal(0, 0.2, 2*LSTM_HIDDEN_SIZE).astype(np.float32).reshape((1,10))
  state = data[0]

  # Run through data and update in batches of BUFFER_SIZE.
  for t in xrange(1, len(data)):
    if t % 1000 == 0:
      print t

    qvals_eval, new_lstm_state_eval = \
        sess.run([qvals, new_lstm_state], feed_dict={inputs:state.reshape((1,5)), lstm_state:lstate})

    # Choose action.
    if (random.random() < epsilon):
      action = random.choice(ACTION_LIST)
    else: 
      action = np.argmax(qvals_eval)
    
    # Observe reward. 
    reward = get_reward(action, t, data)
    current_lstm_state = new_lstm_state_eval

    # Increment state.
    new_state = data[t:t+1]
    action_history.append(action)

    # Store action.
    cur_replay_step = (state, action, reward, new_state, qvals_eval, lstm_state, current_lstm_state)
    if (len(replay_buffer) < BUFFER_SIZE):
      replay_buffer.append(cur_replay_step)
    else:

      X_train = []
      y_train = []
      for old_state, action, reward, new_state, old_qvals, old_lstate, new_lstate in replay_buffer:
        # Get max_Q(S',a)
        #sess.run(qvals, feed_dict={inputs=old_state, lstm_state=old_lstate})
        #old_qvals = qvals
        qvals_eval = sess.run(qvals, feed_dict={inputs:new_state.reshape((1,5)), lstm_state:new_lstate})
        new_qvals = qvals_eval
        max_qval = np.max(new_qvals)
        # Q-update
        update = old_qvals[action] + alpha * (reward + gamma * max_qval - old_qvals[action])
        y = np.zeros((NUM_ACTIONS, 1))
        y[:] = old_qvals[:]
        y[action] = update

        X_train.append(old_state)
        y_train.append(y)

      # Update.
      for j in xrange(len(X_train)):
        x = X_train[j]
        y = y_train[j]
        cur_lstate = replay_buffer[j][-1]
        sess.run(train_step, feed_dict={inputs:x.reshape((1,5)), gold:y.reshape((1,3)), lstm_state:cur_lstate.reshape((1,10))})

      # Reset.
      replay_buffer = []

    
    state = new_state
    lstate = current_lstm_state
    #print "current_lstm_state type", type(current_lstm_state)

 
  # Update exploration factor.
  if epsilon > 0.1:
    epsilon -= (1.0 / EPOCHS)

  
  reward_sum = max_reward(sess, data) 
  print "Epoch #: {}\tReward {}".format(k, reward_sum) 

# Test.






