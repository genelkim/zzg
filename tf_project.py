
import numpy as np
np.random.seed(1335) # for reproducibility
np.set_printoptions(precision=5, suppress=True, linewidth=150)

from matplotlib import pyplot as plt


import quandl


# Mean Squared Error Loss Function for TensorFlow.
def mse_loss_fn(prediction, gold):
  return tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(prediction, gold), 2)))

# Index of data for price(close)
PRICE_INDEX = 1

# Reward is difference in price(close) difference * \
#           action (number of shares sold/bought)
def get_reward(action, t, data):
  return (data[t-1][PRICE_INDEX] - data[t-2][PRICE_INDEX])*action
  












# Data format:
# Open, Close, Low, High, Volume
data = np.genfromtxt('data/sorted_data.csv', delimiter=",")[1:,2:]


#
# Global Parameters.
#

BUY_ACT = 100
SELL_ACT = -100
HOLD_ACT = 0
ACTION_LIST = [SELL_ACT, HOLD_ACT, BUY_ACT]
NUM_ACTIONS = len(ACTION_LIST)


#
# LSTM Definition
#

BATCH_SIZE = 200
NUM_FEATURES = 5

STEPSIZE = 1 

LSTM_HIDDEN_SIZE = 64

# TODO: fill
# Internal Representations.  TensorFlow keeps track of values between
# steps in an interactive session.
#W = tf.Variable(...) # weights
#b = tf.Variable(...) # bias
#state = np.zeros((1,NUM_FEATURES))
# num_actions x lstm_hidden_size matrix of weights.
FW = tf.Variable(tf.truncated_normal([NUM_ACTIONS, LSTM_HIDDEN_SIZE], stddev=0.2, mean=0, dtype=tf.float32), name="FW", dtype=tf.float32)

# Placeholders for dynamic inputs.
inputs = tf.placeholder(tf.float32, shape=(BATCH_SIZE,1))
lstm_state = tf.placeholder(tf.float32, shape=(LSTM_HIDDEN_SIZE))
gold = tf.placeholder(tf.float32, shape=(BATCH_SIZE,1))


lstm = tf.contrib.rnn.BasicLSTMCell(LSTM_HIDDEN_SIZE)
# TODO: add dropout 0.5

#initial_state = tf.zeros([BATCH_SIZE, lstm.state_size])
#state = initial_state

outputs, new_lstm_state = lstm(inputs, lstm_state)
qvals = tf.matmul(FW, outputs)

# TODO: prediction?
#prediction = tf.nn.softmax(tf.matmul(outputs, W) + b)
loss = mse_loss_fn(qvals, gold)
train_step = tf.train.AdamOptimizer(STEPSIZE).minimize(loss)

# Start interactive session.
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


#
# Q-Learning (Main).
# 


# Q-Learning Parameters
gamma = 0.95  # discount factor
alpha = 1     # learning rate
epsilon = 1   # exploration factor

# Global Parameters
EPOCHS = 10

replay_buffer = []
action_history = []

WINDOW_SIZE = 15

# TODO: do we need this?
input_data, close_data = format_data(raw_data)

# Train.
for k in range(epochs):
  # Initialize state and data.
  lstm_state = np.random.normal(0, 0.2, LSTM_HIDDEN_SIZE)
  state = data[0]

  # Run through data and update in batches of BATCH_SIZE.
  for t in xrange(1, len(data)):
    sess.run(qvals, feed_dict={inputs=state, lstm_state=lstm_state})

    # Choose action.
    if (random.random() < epsilon):
      action = random.choice(ACTION_LIST)
    else: 
      action = np.argmax(qvals)

    # Increment state.
    new_state = data[t:t+1]
    action_history.append(action)

    # Observe reward. 
    reward = get_reward(action, t, data)

    # Replay actions.
    cur_replay_step = (state, action, reward, new_state, lstm_state, new_lstm_state, qvals)
    if (len(replay_buffer) < BATCH_SIZE):
      replay_buffer.append(cur_replay_step)
    else:

      X_train = []
      y_train = []
      for old_state, action, reward, new_state, old_lstate, new_lstate, old_qvals in replay_buffer:
        # Get max_Q(S',a)
        #sess.run(qvals, feed_dict={inputs=old_state, lstm_state=old_lstate})
        #old_qvals = qvals
        sess.run(qvals, feed_dict={inputs=new_state, lstm_state=new_lstate})
        new_qvals = qvals
        max_qval = np.max(new_qvals)
        # Q-update
        update = old_qvals[action] + alpha * (reward + gamma * max_qval - old_qvals[action])
        y[:] = old_qvals[:]
        y[action] = update

        X_train.append(old_state)
        y_train.append(y)

      # Update.
      for j in xrange(len(X_train)):
        x = X_train[j]
        y = y_train[j]
        lstate = replay_buffer[j][-1]
        sess.run(train_step, feed_dict={input=x, gold=y, lstm_state=lstate})

      # Reset.
      replay_buffer = []

    
    state = new_state
    lstm_state = new_lstm_state



# Test.






