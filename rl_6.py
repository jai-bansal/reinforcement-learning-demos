# DEEP RECURRENT Q-NETWORK
# This script implements a Deep Recurrent Q-Network (DRQN) which can solve Partially Observable Markov Decision Processes (POMDP). 
# Associated blog post: https://medium.com/p/68463e9aeefc

# This repo has more reinforcment learning tutorials and the additional required gridworld.py and helper.py files:
# https://github.com/awjuliani/DeepRL-Agents

##################
# IMPORT LIBRARIES
##################

import numpy as np
import random
import tensorflow as tf
#import matplotlib.pyplot as plt
#import scipy.misc
import os
import csv
#import itertools
import tensorflow.contrib.slim as slim

# Make sure to download repo (https://github.com/awjuliani/DeepRL-Agents) and have "helper.py" file in working directory.
# "helper.py" contains helper functions.
# Don't fully get "helper.py"...see that file for more info
from helper import *

from gridworld import gameEnv

#######################
# LOAD GAME ENVIRONMENT
#######################

# Gridworld size can be adjusted. Making it smaller/larger (adjusting size) makes task easier/harder.

# Setting "partial" = "True" limits the field of view, resulting in a 
# partially observable MDP. Setting "partial" = "False" provides the agent 
# with the entire environment, resulting in a fully observable MDP.

env = gameEnv(partial = False, size = 9)
env = gameEnv(partial = True, size = 9)

# Above are examples of starting environments. Agent controls the blue square, and can move 
# up/down/left/right. Goal is to move to green squares (+1 reward) and avoid red squares (-1 reward). 
# When the agent moves through a green or red square, the square is randomly moved to a new place.

##########################
# IMPLEMENTING THE NETWORK
##########################

class Qnetwork():
    
    # "h_size" is size of final conv layer output.
    # "rnn_cell" will be an LSTM cell. "myScope" is used for naming variables.
    def __init__(self, h_size, rnn_cell, myScope):

        # Network receives a game frame flattened into an array.
        # It resizes the frame and processes it through 4 conv layers.
        
        self.scalarInput = tf.placeholder(shape = [None, 21168], dtype = tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape = [-1, 84, 84, 3])

        self.conv1 = slim.convolution2d(inputs = self.imageIn, 
                                        num_outputs = 32, 
                                        kernel_size = [8, 8], 
                                        stride = [4, 4], 
                                        padding = 'VALID', 
                                        biases_initializer = None, 
                                        scope = myScope + '_conv1')
        self.conv2 = slim.convolution2d(inputs = self.conv1, 
                                        num_outputs = 64,
                                        kernel_size = [4, 4], 
                                        stride = [2, 2], 
                                        padding = 'VALID', 
                                        biases_initializer = None,
                                        scope = myScope + '_conv2')
        self.conv3 = slim.convolution2d(inputs = self.conv2,
                                        num_outputs = 64, 
                                        kernel_size = [3, 3], 
                                        stride = [1, 1], 
                                        padding = 'VALID',
                                        biases_initializer = None,
                                        scope = myScope + '_conv3')
        self.conv4 = slim.convolution2d(inputs = self.conv3, 
                                        num_outputs = h_size, 
                                        kernel_size = [7, 7], 
                                        stride = [1, 1], 
                                        padding = 'VALID', 
                                        biases_initializer = None,
                                        scope = myScope + '_conv4')
        
        # We use consecutive frames to understand temporal dependencies. This is the # of consecutive frames.
        self.trainLength = tf.placeholder(dtype = tf.int32)

        # Final conv layer output is sent to recurrent layer.
        # A trace is a sequence of experiences from within an episode.
        # We use consecutive frames to understand temporal dependencies.
        # Input must be reshaped into [batch x trace x units] for RNN processing, 
        # and then returned to [batch x units*] when sent through the upper levels.
        # Reshaping must be done because that's what the "tf.nn.dynamic_rnn" function accepts.
        # I guess we change back so each frame can be evaluated for value and action computation.
        
        self.batch_size = tf.placeholder(dtype = tf.int32, shape = [])
        
        # "flatten" converts "conv4" from (?, 1, 1, h_size) to (?, h_size).
        # Reshaping must be done because that's what the "tf.nn.dynamic_rnn" function accepts.
        self.convFlat = tf.reshape(slim.flatten(self.conv4), 
                                   [self.batch_size, self.trainLength, h_size])   
        
        # Return zero-filled state tensor (initial state)
        # "state_in" is actually 2 Tensors. I think one is for the output and the other is for the hidden state.
        self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32) 
        
        # Feed "self.convFlat" into "rnn_cell". "rnn_cell" has initial state "self.state_in".
        # "self.rnn" / "self.rnn_state" is RNN output / final state.
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs = self.convFlat, 
                                                     cell = rnn_cell, 
                                                     dtype = tf.float32, 
                                                     initial_state = self.state_in, 
                                                     scope = myScope + '_rnn')
        
        # I guess we change back so each frame can be evaluated for value and action computation
        self.rnn = tf.reshape(self.rnn, shape = [-1, h_size]) 
        
        # Split recurrent layer output into value and advantage streams
        self.streamA, self.streamV = tf.split(self.rnn, 2, 1)
        
        self.AW = tf.Variable(tf.random_normal([h_size // 2, 4])) # Create variables w/ values initialized by random normal distribution
        self.VW = tf.Variable(tf.random_normal([h_size // 2, 1]))
        
        # "streamA"/"streamV" are relevant frame + RNN info to find action/state value.
        # Multiply them w/ weights "AW"/"VW" to get (predicted) action/state value.
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)
        
        # Derivative of sum of "Advantage" wrt to each pixel in each image in "imageIn".
        # This is the only mention of "salience" in this script! See "helper.py" for more
        # "salience" has dimensions: (?, 84, 84, 3)
        # "salience" gives an idea of what pixels contribute the most to changing "Advantage".
        # A large gradient for a pixel indicates that pixel changing would change "Advantage" a lot.
        # A 0 gradient for a pixel indicates that pixel doesn't really matter for "Advantage".
        # See "https://raghakot.github.io/keras-vis/visualizations/saliency/" for more info
        self.salience = tf.gradients(self.Advantage, self.imageIn)
        
        # Get final Q-values. Add state value to relative action advantage.
        # For each action, subtract the average action value. That leaves the relative value of each action.
        self.Qout = self.Value + tf.subtract(self.Advantage, 
                                             tf.reduce_mean(self.Advantage, axis = 1, keepdims = True))
        self.predict = tf.argmax(self.Qout, 1)
        
        # Get loss by taking sum of squares difference between target and predicted Q-values
        self.targetQ = tf.placeholder(shape = [None], dtype = tf.float32)
        self.actions = tf.placeholder(shape = [None], dtype = tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, 4, dtype = tf.float32)
        
        # These are predicted Q-values for chosen actions. 
        # The inner "multiply" keeps the Q-values only for actions that were taken, because of the "actions_onehot" term.
        # "reduce_sum" reduces dimensions by removing zero terms.
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis = 1)
        
        self.td_error = tf.square(self.targetQ - self.Q)
        
        # To only propogate accurate gradients through the network, mask 1st half of 
        # losses for each trace as per Lample & Chatlot 2016.
        # A trace is a sequence of experiences from within an episode.
        # We use consecutive frames to understand temporal dependencies.
        # Research has shown only sending the last half of gradients improves performance 
        # by only sending more meaningful info through the network.
        # Reminds me of dropout.
        # See blog post for more info: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-6-partial-observability-and-deep-recurrent-q-68463e9aeefc
        self.maskA = tf.zeros([self.batch_size, self.trainLength // 2])
        self.maskB = tf.ones([self.batch_size, self.trainLength // 2])
        self.mask = tf.concat([self.maskA, self.maskB], 1)
        self.mask = tf.reshape(self.mask, [-1])                  # Make mask 1D
        self.loss = tf.reduce_mean(self.td_error * self.mask)
        
        self.trainer = tf.train.AdamOptimizer(learning_rate = 0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

###################
# EXPERIENCE REPLAY
###################
# Create classes to store experiences and sample them randomly to train the network. 
# The episode buffer stores experiences for a single episode. 
# The experience buffer stores entire episodes of experiences. 
# "sample()" samples training batches.
        
class experience_buffer():
    def __init__(self, buffer_size = 1000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):   # Function to add new "experience" to "buffer"
        
        # If adding new "experience" to "buffer" makes "buffer" too big (larger than "buffer_size"), 
        # remove an experience from front of "buffer" to make room.
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0 : (1 + len(self.buffer)) - self.buffer_size] = []
            
        self.buffer.append(experience)
        
    # Sample some traces. Traces have length "trace_length".
    # A trace is a sequence of experiences from within an episode. We use consecutive frames to understand temporal dependencies.
    def sample(self, batch_size, trace_length):
        
        # Below, each element of "buffer" will be an entire episode
        sampled_episodes = random.sample(self.buffer, batch_size)
        sampledTraces = []
        
        for episode in sampled_episodes:     # Select 1 trace from each episode
            
            # Randomly pick starting point of trace.
            # The 2nd argument is ONE ABOVE the largest integer that can be drawn.
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            
            # Pick "trace_length" consecutive experiences from an episode
            sampledTraces.append(episode[point : point + trace_length])
            
        sampledTraces = np.array(sampledTraces)
        
        # Experiences consist of original state, action, reward, new state, and done status
        return np.reshape(sampledTraces, [batch_size * trace_length, 5])
    
#########################
# SET TRAINING PARAMETERS
#########################
batch_size = 4       # How many experience traces to use for each training step
trace_length = 8              # Length of each experience trace during training
update_freq = 5                                   # Frequency of training steps
y = 0.99                         # Discount factor for Q-values of "next" state   
startE = 1                                  # Initial random action probability
endE = 0.1                                    # Final random action probability   
annealing_steps = 10000 # Number of training steps to reduce "startE" to "endE"
num_episodes = 800 # Number of game episodes to train network with (originally 10000)
pre_train_steps = 10000  # Number of random action steps before training begins
load_model = False
path = "./rl_6_model"                                      # Path to save model
h_size = 512                                  # Size of final conv layer output
max_epLength = 25           # Maximum allowed length of episode (originally 50)
time_per_step = 1                          # Length of each step in output GIFs
summaryLength = 100       # Number of episodes to periodically use for analysis
tau = 0.001 # Rate to update target network parameters toward primary network parameters

######################
# TRAINING THE NETWORK
######################

tf.reset_default_graph()                           # Reset global default graph

# Define cells for primary and target Q-networks.
# I assume "num_units" = "h_size" is configured to allow multiplication w/ the previous input.
cell = tf.contrib.rnn.BasicLSTMCell(num_units = h_size, state_is_tuple = True)
cellT = tf.contrib.rnn.BasicLSTMCell(num_units = h_size, state_is_tuple = True)

mainQN = Qnetwork(h_size, cell, 'main')                     # Create Q-networks
targetQN = Qnetwork(h_size, cellT, 'target')
    
init = tf.global_variables_initializer() # Returns op that initializes global variables
saver = tf.train.Saver() # Creates op to save and restore variables to/from checkpoints

trainables = tf.trainable_variables()    
    
# Returns operation to update target network parameters towards primary network parameters
targetOps = updateTargetGraph(trainables, tau)

myBuffer = experience_buffer()                       # Create experience buffer
    
# Set rate of random action probability decrease 
e = startE                                     # Set initial random action rate
stepDrop = (startE - endE) / annealing_steps

jList = []                                          # List of steps per episode
rList = []                                  # List of total rewards per episode
total_steps = 0                           # Total steps taken across everything

if not os.path.exists(path):                        # Create path to save model
    os.makedirs(path)

with open('rl_6_model/log.csv', 'w') as myfile:         # Write 1st line of the master log-file for Control Center
    wr = csv.writer(myfile, quoting = csv.QUOTE_ALL)
    wr.writerow(['Episode', 'Length', 'Reward', 'IMG', 'LOG', 'SAL'])  

with tf.Session() as sess:
    
    if load_model == True:                       # "load_model" is "False" here
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        
    sess.run(init)
    
    # This is unintuitive structure to me, but is used to run "updateTargetGraph". 
    # "updateTargetGraph" is used to move target network parameters towards primary network parameters.
    # Don't fully understand this in "helper.py"...see that file for more details.
    # I think the original comment below is wrong.
    updateTarget(targetOps, sess) # Set target network to be equal to primary network [ORIGINAL COMMENT]

    for i in range(num_episodes):
        
        episodeBuffer = []
        
        sP = env.reset()            # Reset environment and get 1st observation
        s = processState(sP)   # Function to reshape game frames in "helper.py"
        
        d = False                                                 # Done status
        rAll = 0                                       # Episode reward tracker
        j = 0                                            # Episode step tracker
        
        # State consists of RNN output and memory.
        # Initially, there's nothing to remember so a memory made up of zeros makes sense.
        state = (np.zeros([1, h_size]), np.zeros([1, h_size])) # Reset recurrent layer's hidden state
        
        # The Q-Network
        while j < max_epLength: # Trial ends when episode steps hit max episode steps
            j+=1

            # Pick action
            
            # Pick random action w/ "e" chance. Definitely pick random action for the 1st "pre_train_steps" steps.
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                
                # Update network state (even though action is picked randomly).
                # We're not training here, so I guess "trainLength" and "batch_size" can be 1.
                # There's no placeholder for "state_in", but we don't want to keep using the initial all zeros state.
                # So we use the most updated state. Didn't know you could assign non-placeholders like this...
                state1 = sess.run(mainQN.rnn_state,
                                  feed_dict = {mainQN.scalarInput: [s / 255.0], 
                                               mainQN.trainLength: 1, 
                                               mainQN.state_in: state,
                                               mainQN.batch_size: 1})
    
                a = np.random.randint(0, 4)              # Pick action randomly
                
            else:
                
                # "predict" is best action as picked by network. "rnn_state" updates network state.
                # We're not training here, so I guess "trainLength" and "batch_size" can be 1.
                # There's no placeholder for "state_in", but we don't want to keep using the initial all zeros state.
                # So we use the most updated state. Didn't know you could assign non-placeholders like this...
                a, state1 = sess.run([mainQN.predict, mainQN.rnn_state],
                                     feed_dict = {mainQN.scalarInput: [s / 255.0], 
                                                  mainQN.trainLength: 1, 
                                                  mainQN.state_in: state,
                                                  mainQN.batch_size: 1})
    
                a = a[0]
                
            s1P, r, d = env.step(a) # Take action "a" and get new state, reward, and done status
            s1 = processState(s1P)                          # Reshape new state
            total_steps += 1

            # Save experience to episode buffer. Experience consists of original state, action, reward, new state, and done status.
            episodeBuffer.append(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))
            
            # Only use random actions for the 1st "pre_train_steps" steps.
            # After that, random action probability "e" decreases by "stepDrop" until it hits "endE".
            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop

                if total_steps % (update_freq) == 0:                    
                
                    # This is unintuitive structure to me, but is used to run "updateTargetGraph". 
                    # "updateTargetGraph" is used to move target network parameters towards primary network parameters.
                    # Don't fully understand this in "helper.py"...see that file for more details.
                    updateTarget(targetOps, sess)
                    
                    # Reset recurrent layer's hidden state. Training batches are independent, so no reason to keep old memory.
                    state_train = (np.zeros([batch_size, h_size]), np.zeros([batch_size, h_size])) 
                    
                    trainBatch = myBuffer.sample(batch_size, trace_length) # Get random batch of experiences
  
                    # Perform Double-DQN update to target Q-values
                    
                    # Output are actions picked by MAIN network. "scalarInput" are result states.
                    # There's no placeholder for "state_in", but we use the one from above.
                    # Didn't know you could assign non-placeholders like this...
                    Q1 = sess.run(mainQN.predict, 
                                  feed_dict = {mainQN.scalarInput: np.vstack(trainBatch[:, 3] / 255.0), 
                                               mainQN.trainLength: trace_length, 
                                               mainQN.state_in: state_train,
                                               mainQN.batch_size: batch_size})
                    
                    # Output are vectors of 4 Q-values (1 per action) from TARGET network.
                    # There's no placeholder for "state_in", but we use the one from above.
                    # Didn't know you could assign non-placeholders like this...
                    Q2 = sess.run(targetQN.Qout, 
                                  feed_dict = {targetQN.scalarInput: np.vstack(trainBatch[:, 3] / 255.0), 
                                               targetQN.trainLength: trace_length, 
                                               targetQN.state_in: state_train,
                                               targetQN.batch_size: batch_size})

                    # 4th index of "trainBatch" is done status: "True" or "False"
                    # These correspond to 1 or 0 respectively.
                    # Result: "True"/"False" becomes 0/1 (flipped).
                    # This will be a multiplier for target Q-values.
                    end_multiplier = -(trainBatch[:, 4] - 1)
                    
                    # "Q2" are vectors of 4 Q-values (1 per action) from TARGET network.
                    # "Q1" are action indices chosen by MAIN network ("mainQN").
                    # Result: "batch_size * trace_length" Q-values each corresponding to an action chosen by main network.
                    doubleQ = Q2[range(batch_size * trace_length), Q1]
                    
                    # 2nd index of "trainBatch" is reward.
                    # Target Q-value is immediate reward + discounted Q-value looking out from next state.
                    # Because of "end_multiplier", 2nd term is 0 if a move ends the game and 1 otherwise.
                    # So, if a move ends the game, 2nd term is 0 b/c there's no Q-value looking out from the done state.
                    targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)
       
                    # Update MAIN network using target Q-values.
                    # There's no placeholder for "state_in", but we use the one from above.
                    # Didn't know you could assign non-placeholders like this...
                    sess.run(mainQN.updateModel, 
                             feed_dict = {mainQN.scalarInput: np.vstack(trainBatch[:, 0] / 255.0), 
                                          mainQN.targetQ: targetQ, 
                                          mainQN.actions: trainBatch[:, 1], 
                                          mainQN.trainLength: trace_length, 
                                          mainQN.state_in: state_train,
                                          mainQN.batch_size: batch_size})

            rAll += r                    # Add reward to episode reward tracker
            s = s1                                               # Update state
            #sP = s1P # Don't think this is necessary..."s1P" is the unprocessed version of "s1"
            state = state1                                   # Update RNN state
            
            if d == True:              # If episode is done, break episode loop
                break

        # Add episode to experience buffer
        bufferArray = np.array(episodeBuffer) # "episodeBuffer" contains all experiences in the episode
        episodeBuffer = list(zip(bufferArray))       # Manipulate "bufferArray"
        myBuffer.add(episodeBuffer)          # "myBuffer" holds entire episodes
        jList.append(j)             # Add number of steps in episode to tracker
        rList.append(rAll)                      # Add episode reward to tracker

        if i % 1000 == 0 and i != 0:                  # Periodically save model
            
            saver.save(sess, path + '/model-' + str(i) + '.cptk')
            print ("Saved Model")

        if len(rList) % summaryLength == 0 and len(rList) != 0: # Periodically print info
            
            print("Total Steps:", total_steps,            
                  "Episodes:", i,
                  "Avg. Recent Reward:", np.mean(rList[-summaryLength:]), 
                  "Random Action Prob:", e)
            print('')
            
            # Write out GIFs and CSVs for tracking
            saveToCenter(i, rList, jList,                     # Don't fully get this...see "helper.py" for more info
                         np.reshape(np.array(episodeBuffer), 
                                    [len(episodeBuffer), 5]), 
                         summaryLength, h_size, sess, mainQN, time_per_step)
                         
            #holder = saveToCenter(i, rList, jList, 
            #             np.reshape(np.array(episodeBuffer), 
            #                        [len(episodeBuffer), 5]), 
            #             summaryLength, h_size, sess, mainQN, time_per_step)
            
    saver.save(sess, path + '/model-' + str(i) + '.cptk')    # Save final model
    
#####################
# TESTING THE NETWORK
#####################
    
## REVIEW DONE TO HERE
    
# Parameters
e = 0.01               # Probability of choosing random action (same as "endE")
num_episodes = 8000             # Number of game episodes to test network with
load_model = True                                            # Load saved model
path = "./rl_6_model"                                           # Path to model
h_size = 512                                  # Size of final conv layer output
max_epLength = 25                                      # Maximum episode length
time_per_step = 1                          # Length of each step in output GIFs
summaryLength = 100       # Number of episodes to periodically use for analysis

tf.reset_default_graph()                           # Reset global default graph

# Define cells for primary and target Q-networks.
# I assume "num_units" = "h_size" is configured to allow multiplication w/ the previous input.
cell = tf.contrib.rnn.BasicLSTMCell(num_units = h_size, state_is_tuple = True)
cellT = tf.contrib.rnn.BasicLSTMCell(num_units = h_size, state_is_tuple = True)

mainQN = Qnetwork(h_size, cell, 'main')                     # Create Q-networks
targetQN = Qnetwork(h_size, cellT, 'target')

init = tf.global_variables_initializer() # Returns op that initializes global variables

saver = tf.train.Saver(max_to_keep = 2) # Creates op to save and restore variables to/from checkpoints

jList = []                                          # List of steps per episode
rList = []                                  # List of total rewards per episode
total_steps = 0                           # Total steps taken across everything

if not os.path.exists(path):                        # Create path to save model
    os.makedirs(path)

with open('rl_6_model/log.csv', 'w') as myfile:      # Write 1st line of the master log-file for Control Center
    wr = csv.writer(myfile, quoting = csv.QUOTE_ALL)
    wr.writerow(['Episode', 'Length', 'Reward', 'IMG', 'LOG', 'SAL'])    
    
    #wr = csv.writer(open('./Center/log.csv', 'a'), quoting=csv.QUOTE_ALL) # I'm ignoring this

with tf.Session() as sess:
    
    if load_model == True:                        # "load_model" is "True" here
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        
    else:
        sess.run(init)

    for i in range(num_episodes):
        
        episodeBuffer = []
        
        sP = env.reset()            # Reset environment and get 1st observation
        s = processState(sP)   # Function to reshape game frames in "helper.py"
        
        d = False                                                 # Done status
        rAll = 0                                       # Episode reward tracker
        j = 0                                            # Episode step tracker
        
        # State consists of RNN output and memory.
        # Initially, there's nothing to remember so a memory made up of zeros makes sense.
        state = (np.zeros([1, h_size]), np.zeros([1, h_size])) # Reset recurrent layer's hidden state

        # The Q-Network
        while j < max_epLength: # Trial ends when episode steps hit max episode steps
            j += 1

            # Pick action
            
            if np.random.rand(1) < e:        # Pick random action w/ "e" chance
                
                # Even though we're picking an action randomly, update network state.
                # We're not training here, so I guess "trainLength" and "batch_size" can be 1.
                # There's no placeholder for "state_in", but we don't want to keep using the initial all zeros state.
                # So we use the most updated state. Didn't know you could assign non-placeholders like this...
                state1 = sess.run(mainQN.rnn_state,
                                  feed_dict = {mainQN.scalarInput: [s / 255.0], 
                                               mainQN.trainLength: 1, 
                                               mainQN.state_in: state,   
                                               mainQN.batch_size: 1})
    
                a = np.random.randint(0, 4)              # Pick action randomly
                
            else:
                
                # "predict" is best action as picked by network. "rnn_state" updates network state.
                # We're not training here, so I guess "trainLength" and "batch_size" can be 1.
                # There's no placeholder for "state_in", but we don't want to keep using the initial all zeros state.
                # So we use the most updated state. Didn't know you could assign non-placeholders like this...
                a, state1 = sess.run([mainQN.predict, mainQN.rnn_state],
                                     feed_dict = {mainQN.scalarInput: [s / 255.0], 
                                                  mainQN.trainLength: 1,
                                                  mainQN.state_in: state,
                                                  mainQN.batch_size: 1})
                a = a[0]

            s1P, r, d = env.step(a) # Take action "a" and get new state, reward, and done status
            s1 = processState(s1P)                          # Reshape new state
            total_steps += 1

            # Save experience to episode buffer. Experience consists of original state, action, reward, new state, and done status.
            episodeBuffer.append(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))
            
            # No training update step here

            rAll += r                    # Add reward to episode reward tracker
            s = s1                                               # Update state
            #sP = s1P # Don't think this is necessary..."s1P" is the unprocessed version of "s1"
            state = state1                                   # Update RNN state          
            
            if d == True:              # If episode is done, break episode loop
                break
            
        bufferArray = np.array(episodeBuffer) # "episodeBuffer" contains all experiences in the episode
        jList.append(j)             # Add number of steps in episode to tracker
        rList.append(rAll)                      # Add episode reward to tracker

        if len(rList) % summaryLength == 0 and len(rList) != 0: # Periodically print info
            
            print("Total Steps:", total_steps,                  
                  "Episodes:", i,
                  "Avg. Recent Reward:", np.mean(rList[-summaryLength:]), 
                  "Random Action Prob:", e)
            print('')
            
            # Write out GIFs and CSVs for tracking
            saveToCenter(i, rList, jList,                      # Don't fully get this...see "helper.py" for more info
                         np.reshape(np.array(episodeBuffer), 
                                    [len(episodeBuffer), 5]),
                summaryLength, h_size, sess, mainQN, time_per_step)

# This probably won't change much when testing...I'm not updating the network anymore.
sum(rList) / len(rList)                            # Average reward per episode
sum(rList[0:100]) / len(rList[0:100])  # Average reward over first 100 episodes
sum(rList[-100:]) / len(rList[-100:])   # Average reward over last 100 episodes