from __future__ import division
import numpy as np
import numpy.random as random
#random.seed(1335) # for reproducibility
#np.set_printoptions(precision=5, suppress=True, linewidth=150)

from matplotlib import pyplot as plt
import tensorflow as tf
from collections import namedtuple

AgentState = namedtuple("AgentState", ["cash", "stocks"])
INITIAL_AGENT_STATE = AgentState(60000, 100) # Cash/stocks
RewardHistory = namedtuple("RewardHistory", ["single", "window", "final"])
#
# Load Data
#

#datafile = 'data/dummydata.3000'
datafile = 'data/linear_dummy_noise_50_100.csv'
# Index of data for price(close)
PRICE_INDEX = 0


# Data format:
# Open, Close, Low, High, Volume
#data = np.genfromtxt('data/sorted_data.csv.10000', delimiter=",")[1:,2:]
# Data subset
#data = data[:10000]
#NUM_FEATURES = 5
raw_data = np.genfromtxt(datafile, delimiter=",")
raw_data = raw_data.reshape((len(raw_data), 1))

# Rescale data.
means = np.mean(raw_data, axis=0)
variances = np.var(raw_data, axis=0)
normed_data = [x - means for x in raw_data]
normed_data = np.array([x / variances for x in normed_data])
# Placeholder for qstate values.
qstate_zeros = np.zeros((raw_data.shape[0], len(INITIAL_AGENT_STATE))) 

data = np.append(raw_data, normed_data, axis=1)
data = np.append(data, qstate_zeros, axis=1)


#
# Global Parameters.
#

# Actions.
BUY_ACT = 2
SELL_ACT = 0
HOLD_ACT = 1
ACTION_LIST = [SELL_ACT, HOLD_ACT, BUY_ACT]
NUM_ACTIONS = len(ACTION_LIST)
ACTION_VAL = { BUY_ACT : 1, SELL_ACT : -1, HOLD_ACT : 0.0 }

# Q-Learning.
EPOCHS = 20
BUFFER_SIZE = 200
# Gene: I set discout factor to 0 since our current action don't actually cause a state transition.
gamma = 1.0/EPOCHS  # discount factor
#gamma = 0
alpha = 1     # learning rate
epsilon = 1   # exploration factor
WINDOW_SIZE = 15

FINAL_REWARD_FACTOR = 1.0
WINDOW_REWARD_FACTOR = 1.0 / ((len(data) / WINDOW_SIZE) * 10)
SINGLE_REWARD_FACTOR = 1.0 / (len(data) * 100)

# LSTM.
NUM_FEATURES = data.shape[1]
LSTM_STATE_SIZE = NUM_FEATURES
FULL_LSTM_STATE_SIZE = 2*LSTM_STATE_SIZE

NUM_FEATURES_EXPAND = 2*NUM_FEATURES
LSTM_STATE_SIZE_HIDDEN = NUM_FEATURES_EXPAND
FULL_LSTM_STATE_SIZE_HIDDEN = 2*LSTM_STATE_SIZE_HIDDEN


BATCH_SIZE = 1
STEPSIZE = 1 


LSTM_INIT = np.random.normal(0, 0.2, FULL_LSTM_STATE_SIZE).astype(np.float32).reshape((1,FULL_LSTM_STATE_SIZE))
LSTM_INIT_HIDDEN = np.random.normal(0, 0.2, FULL_LSTM_STATE_SIZE_HIDDEN).astype(np.float32).reshape((1,FULL_LSTM_STATE_SIZE_HIDDEN))


print "Parameters"
print "gamma", gamma
print "alpha", alpha
print "epsilon", epsilon
print "buffer size", BUFFER_SIZE
print "EPOCHS", EPOCHS
print "file", datafile

print ""
print "SHORT UPDATE"
#print "LONG UPDATE"

#
# Helper Functions
#

# Mean Squared Error Loss Function for TensorFlow.
def mse_loss_fn(prediction, gold):
  return tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(tf.transpose(prediction), gold), 2)))

# Python MSE
def mse_loss_py(prediction, gold):
  return np.sqrt(np.sum(np.power(prediction.T - gold, 2)))

# Reward is difference in price(close) difference * \
#           action (number of shares sold/bought)
def simple_reward(action, t, data):
  return (data[t][PRICE_INDEX] - data[t-1][PRICE_INDEX])*ACTION_VAL[action]

# Reward based on agent state of cash + stock count.
# Stock >= 0.
# 3 different types of rewards:
#   1. Final reward (at end of epoch, gets a large reward/cost based off of state)
#   2. Window reward (at the end of each window, a medium reward based off of state)
#   3. Single reward (each step gets a small reward)
def stateful_reward(action, t, data, agent_state, past_reward):
  # Value of current state.
  state_val = agent_state.cash + agent_state.stocks*data[t][PRICE_INDEX]
  if agent_state.stocks < 0:
    state_val = -10e10

  total_reward = 0
  new_reward = list(past_reward)
  # Compute final reward.
  if t == len(data) - 1:
    total_reward += (state_val - past_reward.final)*FINAL_REWARD_FACTOR
    new_reward[2] = state_val
  if t % WINDOW_SIZE == 0:
    total_reward += (state_val - past_reward.window)*WINDOW_REWARD_FACTOR
    new_reward[1] = state_val

  total_reward += (state_val - past_reward.single)*SINGLE_REWARD_FACTOR
  new_reward[0] = state_val

  return total_reward, RewardHistory(new_reward[0], new_reward[1], new_reward[2])
  

# Reward function.
def get_reward(action, t, data, agent_state, past_reward):
  return stateful_reward(action, t, data, agent_state, past_reward)
  

# Evaluate performance on optimal strategy (no epsilon or random action).
def max_reward(sess, data):
  agent_state = INITIAL_AGENT_STATE

  initial_value = agent_state.cash + (agent_state.stocks * data[0][PRICE_INDEX])
  past_reward = RewardHistory(initial_value, initial_value, initial_value)
  
  lstate = LSTM_INIT
  lstate_hidden = LSTM_INIT_HIDDEN
  state = data[0]
  state[-2] = agent_state.cash
  state[-1] = agent_state.stocks
  reward_sum = 0
  total_loss = 0

  actions = []

  # Initial step.
  qvals_eval, new_lstm_state_eval,new_lstm_state_eval_hidden= \
      sess.run([qvals, new_lstm_state, new_lstm_state_hidden], feed_dict={inputs:state.reshape((1,NUM_FEATURES)), lstm_state:lstate, lstm_state_hidden:lstate_hidden})
  action = np.argmax(qvals_eval)
  actions.append(action)
  
  # Store oldvals,
  old_qvals = qvals_eval
  old_state = state
  
  #rewards.append(reward)
  lstate = new_lstm_state_eval
  lstate_hidden = new_lstm_state_eval_hidden

  state = data[1]
  state[-2] = old_state[-2] - ACTION_VAL[action]*old_state[PRICE_INDEX]
  state[-1] = old_state[-1] + ACTION_VAL[action]
  agent_state = AgentState(state[-2], state[-1])
  
  
  reward, past_reward = get_reward(action, 1, data, agent_state, past_reward)
  reward_sum += reward


  # Compute prediction and gold in tandem.
  for t in xrange(2, len(data)):
    # Gold computation.
    qvals_eval, new_lstm_state_eval, new_lstm_state_eval_hidden = \
        sess.run([qvals, new_lstm_state, new_lstm_state_hidden], feed_dict={inputs:state.reshape((1,NUM_FEATURES)), lstm_state:lstate, lstm_state_hidden: lstate_hidden})
    # Compute loss.
    max_qval = np.max(qvals_eval)
    #update = old_qvals[action] + alpha * (reward + gamma * max_qval - old_qvals[action])
    update = alpha * (reward + (gamma * max_qval))
    y = np.zeros((NUM_ACTIONS, 1))
    y[:] = old_qvals[:]
    y[action] = update
    #print old_qvals
    #print y
    loss = mse_loss_py(old_qvals, y) # Seems a bit funny... the loss is always the reward size...
    total_loss += loss

    # Store reward and transition state.
    action = np.argmax(qvals_eval)
    actions.append(action)
    
    old_qvals = qvals_eval
    old_state = state
    
    lstate = new_lstm_state_eval
    lstate_hidden = new_lstm_state_eval_hidden
    state = data[t]
    state[-2] = old_state[-2] - ACTION_VAL[action]*old_state[PRICE_INDEX]
    state[-1] = old_state[-1] + ACTION_VAL[action]
    agent_state = AgentState(state[-2], state[-1])
    
    reward, past_reward = get_reward(action, t, data, agent_state, past_reward)
    reward_sum += reward
    #rewards.append(reward)


  print np.bincount(actions)
  return reward_sum, total_loss


#
# LSTM Definition with Fully Conncted output layer.
#
# Placeholders that are supplied each call:
#   input for current step
#   lstm hidden state from previous step
#   gold value for training
inputs = tf.placeholder(tf.float32, shape=(BATCH_SIZE,NUM_FEATURES), name="tf_inputs")
gold = tf.placeholder(tf.float32, shape=(BATCH_SIZE,NUM_ACTIONS), name="tf_gold")

#Hidden version of input, also the output from last layer
inputs_hidden = tf.placeholder(tf.float32,shape=(BATCH_SIZE,NUM_FEATURES_EXPAND), name="tf_inputs_hidden");

# c - LSTM speak
lstm_state = tf.placeholder(tf.float32, shape=(BATCH_SIZE,FULL_LSTM_STATE_SIZE), name="tf_lstm_state")

#Hidden LSTM speak
lstm_state_hidden = tf.placeholder(tf.float32, shape=(BATCH_SIZE, FULL_LSTM_STATE_SIZE_HIDDEN), name="tf_lstm_state_hidden")

#Fully Connected layer weights and biases.
FW = tf.Variable(tf.truncated_normal([NUM_FEATURES_EXPAND, LSTM_STATE_SIZE], stddev=0.2, mean=0, dtype=tf.float32), \
                 name="FW", dtype=tf.float32)
FB = tf.Variable(tf.truncated_normal([NUM_FEATURES_EXPAND, 1], stddev=0.2, mean=0, dtype=tf.float32),\
                 name="FB", dtype=tf.float32)

#Hidden fully Connected layer weights and biases.
FW_hidden = tf.Variable(tf.truncated_normal([NUM_ACTIONS, LSTM_STATE_SIZE_HIDDEN], stddev=0.2, mean=0, dtype=tf.float32), \
                 name="FW_hidden", dtype=tf.float32)
FB_hidden = tf.Variable(tf.truncated_normal([NUM_ACTIONS, 1], stddev=0.2, mean=0, dtype=tf.float32),\
                 name="FB_hidden", dtype=tf.float32)



# LSTM model (weights are stored implicitly)
with tf.variable_scope('input_layer'):
  lstm = tf.contrib.rnn.BasicLSTMCell(LSTM_STATE_SIZE, state_is_tuple=False)
# Prediction.
#lstm_in = tf.concat([inputs, lstm_state], 0)
  outputs, new_lstm_state = lstm(inputs, lstm_state)

#Output from fully connected layer, this is also the input for hidden lstm layer
inputs_hidden = tf.transpose(tf.matmul(FW, tf.transpose(outputs)))


# TODO: add dropout 0.5

#Hidden LSTM model
with tf.variable_scope('hidden_layer'):
  lstm_hidden = tf.contrib.rnn.BasicLSTMCell(LSTM_STATE_SIZE_HIDDEN, state_is_tuple=False)

#Prediction from hidden layer
  outputs_hidden, new_lstm_state_hidden = lstm_hidden(inputs_hidden, lstm_state_hidden)

#outputs = pred[0]
#new_lstm_state = pred[1]

#Output from hidden fully connected layer
qvals = tf.matmul(FW_hidden, tf.transpose(outputs_hidden))



# Training.
loss = mse_loss_fn(qvals, gold)
#train_step = tf.train.AdamOptimizer(STEPSIZE).minimize(loss)
train_step = tf.train.GradientDescentOptimizer(STEPSIZE).minimize(loss)

# Start interactive session.
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()





#
# Q-Learning (Main).
# 


replay_buffer = []

# Train.
for k in xrange(EPOCHS):
  agent_state = INITIAL_AGENT_STATE

  initial_value = agent_state.cash + (agent_state.stocks * data[0][PRICE_INDEX])
  past_reward = RewardHistory(initial_value, initial_value, initial_value)

  # Initialize state and data.
  lstate = LSTM_INIT
  lstate_hidden = LSTM_INIT_HIDDEN


  state = data[0]
  state[-2] = agent_state.cash
  state[-1] = agent_state.stocks

  # Run through data and update in batches of BUFFER_SIZE.
  for t in xrange(1, len(data)):
    if t % 1000 == 0:
      print t
    qvals_eval, new_lstm_state_eval, new_lstm_state_eval_hidden = \
        sess.run([qvals, new_lstm_state, new_lstm_state_hidden], feed_dict={inputs:state.reshape((1,NUM_FEATURES)), lstm_state:lstate, lstm_state_hidden:lstate_hidden})


    # Choose action.
    if (random.random() < epsilon):
      action = random.choice(ACTION_LIST)
    else: 
      action = np.argmax(qvals_eval)
    
    # Take action, increment state.
    new_state = data[t]
    new_state[-2] = state[-2] - ACTION_VAL[action]*state[PRICE_INDEX]
    new_state[-1] = state[-1] + ACTION_VAL[action]
    agent_state = AgentState(new_state[-2], new_state[-1])

    # Observe reward. 
    reward, past_reward = get_reward(action, t, data, agent_state, past_reward)
    current_lstm_state = new_lstm_state_eval
    current_lstm_state_hidden = new_lstm_state_eval_hidden


    # Store action.
    cur_replay_step = (state, action, reward, new_state, qvals_eval, current_lstm_state, current_lstm_state_hidden)
    if (len(replay_buffer) < BUFFER_SIZE):
      replay_buffer.append(cur_replay_step)
    else:

      X_train = []
      y_train = []
      for old_state, action, reward, new_state, old_qvals, new_lstate, new_lstate_hidden in replay_buffer:
        # Get max_Q(S',a)
        #sess.run(qvals, feed_dict={inputs=old_state, lstm_state=old_lstate})
        #old_qvals = qvals
        qvals_eval = sess.run(qvals, feed_dict={inputs:new_state.reshape((1,NUM_FEATURES)), lstm_state:new_lstate, lstm_state_hidden:new_lstate_hidden})
        max_qval = np.max(qvals_eval)
        # Q-update
        #update = old_qvals[action] + alpha * (reward + gamma * max_qval - old_qvals[action])
        update = alpha * (reward + (gamma * max_qval))
        y = np.zeros((NUM_ACTIONS, 1))
        y[:] = old_qvals[:]
        y[action] = update

        X_train.append(old_state)
        y_train.append(y)

      # Update.
      for j in xrange(len(X_train)):
        x = X_train[j]
        y = y_train[j]
        cur_lstate = replay_buffer[j][-2]
        cur_lstate_hidden = replay_buffer[j][-1]
        sess.run(train_step, \
            feed_dict={inputs:x.reshape((1,NUM_FEATURES)), \
                       gold:y.reshape((1,3)), \
                       lstm_state:cur_lstate.reshape((1,FULL_LSTM_STATE_SIZE)),\
                       lstm_state_hidden:current_lstm_state_hidden.reshape(1,FULL_LSTM_STATE_SIZE_HIDDEN)})

      # Reset.
      replay_buffer = []

    
    state = new_state
    lstate = current_lstm_state
    lstate_hidden = current_lstm_state_hidden
    #print "current_lstm_state type", type(current_lstm_state)

 
  # Update exploration factor.
  if epsilon > 0.1:
    epsilon -= (1.0 / EPOCHS)
  # Update future weight.
  if gamma < 0.95 and gamma > 0.0:
    gamma += (1.0 / EPOCHS)

  
  reward_sum, loss_sum = max_reward(sess, data) 
  print "Epoch #: {}\tReward {}\tLoss {}".format(k, reward_sum, loss_sum) 

# Test.






