#####################################################
# Simple Reinforcement Learning w/ Tensorflow Part 4: 
# Deep Q-Networks and Beyond
#####################################################

# This script implements a Deep Q-Network using Double DQN and Dueling DQN. 
# Agent learns to solve a navigation task in a basic grid world. 
# More info: https://medium.com/p/8438a3e2b8df

# More reinforcment learning tutorials: https://github.com/awjuliani/DeepRL-Agents

# This script is basically the same as: https://github.com/awjuliani/DeepRL-Agents/blob/master/Double-Dueling-DQN.ipynb

# Useful StackExchange link on separate target network vs. double DQN:
# https://datascience.stackexchange.com/questions/32246/q-learning-target-network-vs-double-dqn

################
# IMPORT MODULES
################
from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import scipy.misc
import os

# Make sure to download the repo (https://github.com/awjuliani/DeepRL-Agents) 
# and have "gridworld.py" file in the working directory.
from gridworld import gameEnv

#######################
# LOAD GAME ENVIRONMENT
#######################
# The size of the grid world can be adjusted. 
# Making it smaller/larger provides an easier/harder task for our agent. 
env = gameEnv(partial = False, size = 5)

# More info on the game world: https://github.com/awjuliani/DeepRL-Agents/blob/master/Double-Dueling-DQN.ipynb
# Strangely, game doesn't end when agent gets either 1/-1 reward...it continues.
# So the agent can get multiple rewards in the same episode.

# More info on game world from: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-6-partial-observability-and-deep-recurrent-q-68463e9aeefc
# Goalis to move the blue block to as many green blocks as possible in 50 
# steps while avoiding red blocks. When the blue block moves to a green or 
# red block, that other block is moved to a new random place. Green blocks 
# provide +1 reward, while Red blocks provide -1 reward.

###################
# IMPLEMENT NETWORK
###################

class Qnetwork():
    def __init__(self, h_size): # h_size is the size of final convolutional layer output before splitting into Advantage and Value streams
        
        # Network receives a game frame flattened into an array.
        # It resizes the frame and processes it through 4 convolutional layers.
        
        self.scalarInput =  tf.placeholder(shape = [None, 21168], dtype = tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape = [-1, 84, 84, 3])
        self.conv1 = slim.conv2d(inputs = self.imageIn, 
                                 num_outputs = 32, 
                                 kernel_size = [8, 8],
                                 stride = [4, 4],
                                 padding = 'VALID', 
                                 biases_initializer = None)
        self.conv2 = slim.conv2d(inputs = self.conv1, 
                                 num_outputs = 64, 
                                 kernel_size = [4, 4], 
                                 stride = [2, 2], 
                                 padding = 'VALID', 
                                 biases_initializer = None)
        self.conv3 = slim.conv2d(inputs = self.conv2, 
                                 num_outputs = 64, 
                                 kernel_size = [3, 3], 
                                 stride = [1, 1], 
                                 padding = 'VALID', 
                                 biases_initializer = None)
        self.conv4 = slim.conv2d(inputs = self.conv3, 
                                 num_outputs = h_size,
                                 kernel_size = [7, 7],
                                 stride = [1, 1],
                                 padding = 'VALID', 
                                 biases_initializer = None)  
        
        # Split "conv4" output into separate advantage and value streams.
        self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)
        
        self.streamA = slim.flatten(self.streamAC) # Remove extra "1" dimensions
        self.streamV = slim.flatten(self.streamVC)
        
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([h_size // 2, env.actions])) # Create variables w/ values initialized by "xavier_init"
        self.VW = tf.Variable(xavier_init([h_size // 2, 1]))
        
        # "streamA"/"streamV" are relevant frame info to find action/state value.
        # Multiply them w/ weights "AW"/"VW" to get (predicted) action/state value.
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)   
               
        # Combine to get final Q-values. Add state value to relative action advantage.
        # For each action, subtract the average action value. That leaves the relative value of each action.
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis = 1, keep_dims = True))
        
        self.predict = tf.argmax(self.Qout, 1) # Pick action w/ highest Q-value
        
        # Get loss by taking the sum of squares difference between target and predicted Q-values
        self.targetQ = tf.placeholder(shape = [None], dtype = tf.float32)
        self.actions = tf.placeholder(shape = [None], dtype = tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, env.actions, dtype = tf.float32)
        
        # These are predicted Q-values.
        # The inner "multiply" keeps the Q-values only for actions that were taken, 
        # because of the "actions_onehot" term.
        # "reduce_sum" reduces dimensions by removing zero terms.
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis = 1)
        
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate = 0.0001)
        self.updateModel = self.trainer.minimize(self.loss)
        
###################
# EXPERIENCE REPLAY
###################
# Create class to store experiences and sample them randomly for network training.
# Idea: avoid only using recent experiences for training

class experience_buffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    # Function to add new "experience" to "buffer"
    def add(self, experience):
        
        # If adding new "experience" to "buffer" makes "buffer" too big (larger than "buffer_size"), 
        # remove experiences from the front of "buffer" to make room for new experiences.
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0 : (len(experience) + len(self.buffer)) - self.buffer_size] = []
            
        self.buffer.extend(experience)
            
    # Sample experiences (consisting of original state, action, new state, reward, and done status)
    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])
    
# Resize game frames
def processState(states):
    return np.reshape(states, [21168])

# These functions slowly update target network parameters to primary network parameters.
# Idea: having a target network without constantly shifting Q-values makes training more stable

# Below, variables are structured so the 1st half are from the 
# primary network and the 2nd half are from target network.
# "tau" is the rate to update target network toward primary network.
# Target network parameters slowly move towards primary network parameters.

def updateTargetGraph(tfVars, tau):
    
    total_vars = len(tfVars)
    op_holder = []
    
    for idx, var in enumerate(tfVars[0 : total_vars // 2]): # Loop through 1st half of variables
        
        # Update 2nd half of variables.
        # Assign variable as a combo (based on "tau") of "var" and original variable value.
        op_holder.append(tfVars[idx + total_vars // 2].assign((var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
        
    return(op_holder)

# This is unintuitive structure to me, but is used below to run "updateTargetGraph". 
# "updateTargetGraph" is used to move target network parameters towards primary network parameters.
def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)

######################
# TRAINING THE NETWORK
######################
        
batch_size = 32                          # Experiences to use per training step
update_freq = 4                                   # Frequency of training steps
y = 0.99                         # Discount factor for Q-values of "next" state
startE = 1                               # Initial probability of random action
endE = 0.1                                 # Final probability of random action
annealing_steps = 10000. # Number of training steps to reduce "startE" to "endE"
num_episodes = 1000             # Number of game episodes to train network with
pre_train_steps = 10000 # How many steps of random actions before training begins
max_epLength = 40                                      # Maximum episode length
load_model = False 
path = "./rl_4_model"                                      # Path to save model
h_size = 512 # Size of final convolutional layer output before splitting into Advantage and Value streams
tau = 0.001 # Rate to update target network parameters toward primary network parameters

tf.reset_default_graph()                           # Reset global default graph
mainQN = Qnetwork(h_size)
targetQN = Qnetwork(h_size)

init = tf.global_variables_initializer() # Returns op that initializes global variables

saver = tf.train.Saver() # Creates op to save and restore variables to/from checkpoints

trainables = tf.trainable_variables()

# Returns operation to update target network parameters towards primary network parameters
targetOps = updateTargetGraph(trainables, tau)

myBuffer = experience_buffer()                 # Create overall experience bank

# Set rate of random action probability decrease
e = startE
stepDrop = (startE - endE) / annealing_steps

jList = []                                          # List of steps per episode
rList = []                                  # List of total rewards per episode
total_steps = 0                         # Total steps taken across all episodes

if not os.path.exists(path):                        # Create path to save model
    os.makedirs(path)

with tf.Session() as sess:
    sess.run(init)
    
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        
    for i in range(num_episodes):
        
        episodeBuffer = experience_buffer() # Create experience bank for this episode only

        # Get 1st new observation
        s = env.reset()      # Reset environment (returns an 84 x 84 x 3 frame)
        s = processState(s)   # Reshape frame using user-defined function above

        d = False                                                 # Done status
        rAll = 0                                       # Episode reward tracker
        j = 0                                            # Episode step tracker
        
        # The Q-Network
        while j < max_epLength:     # Trial ends when episode steps hit max episode steps
            j += 1

            # Pick action 
            
            # Pick random action w/ "e" chance.
            # Definitely pick random action for the 1st "pre_train_steps" steps.
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0, 4)
                
            else:                         # Pick action greedily from Q-network
                a = sess.run(mainQN.predict, 
                             feed_dict = {mainQN.scalarInput: [s]})[0]
                
            s1, r, d = env.step(a) # Step environment and get new state, reward, and done status
            s1 = processState(s1)                           # Reshape new state
            total_steps += 1
            
            # Save experience to episode buffer.
            # Experience consists of original state, action, reward, new state, and done status.
            episodeBuffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))
            
            # Only use random actions for the 1st "pre_train_steps" steps.
            # After that, random action probability "e" decreases by "stepDrop" until it hits "endE".
            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop

                if total_steps % (update_freq) == 0:         # Training process
                    
                    trainBatch = myBuffer.sample(batch_size) # Get random batch of experiences

                    # Perform Double-DQN update to target Q-values
                    
                    # "Q1" and "Q2" are used to find the Q-value of the resulting states (after actions).
                    # "feed_dict" are result states. Output are actions picked by MAIN network.
                    Q1 = sess.run(mainQN.predict, 
                                  feed_dict = {mainQN.scalarInput: np.vstack(trainBatch[:, 3])})
                    
                    # "feed_dict" are result states. 
                    # Output are vectors of 4 Q-values (1 per action) from TARGET network.
                    Q2 = sess.run(targetQN.Qout, 
                                  feed_dict = {targetQN.scalarInput: np.vstack(trainBatch[:, 3])})
                    
                    # 4th index of "trainBatch" is done status: "True" or "False"
                    # These correspond to 1 or 0 respectively.
                    # Result: "True"/"False" becomes 0/1 (flipped).
                    # This will be a multiplier for target Q-values.
                    end_multiplier = -(trainBatch[:, 4] - 1)
                    
                     # "Q2" are vectors of 4 Q-values (1 per action) from TARGET network.
                     # "Q1" are action indices chosen by MAIN network ("mainQN").
                     # Result: "batch_size" Q-values each corresponding to an action chosen by main network.
                    doubleQ = Q2[range(batch_size), Q1]
                    
                    # 2nd index of "trainBatch" is reward.
                    # Target Q-value is immediate reward + discounted Q-value looking out from next state.
                    # Because of "end_multiplier", 2nd term is 0 if a move ends the game and 1 otherwise.
                    # So, if a move ends the game, 2nd term is 0 b/c there's no Q-value looking out from the done state.
                    targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)

                    # Update main network weights
                    _ = sess.run(mainQN.updateModel, 
                                 feed_dict = {mainQN.scalarInput: np.vstack(trainBatch[:, 0]), 
                                              mainQN.targetQ: targetQ, 
                                              mainQN.actions: trainBatch[:, 1]})

                    updateTarget(targetOps, sess) # Update target network toward primary network
                    
            rAll += r                    # Add reward to episode reward tracker
            s = s1                                               # Update state
            
            if d == True:              # If episode is done, break episode loop
                break

        # Not in episode loop
        myBuffer.add(episodeBuffer.buffer) # Add episode experiences to overall experience bank
        jList.append(j)            # Add episode steps to overall list of steps
        rList.append(rAll)      # Add episode reward to overall list of rewards

        if i % 500 == 0:                             # Periodically save model
            saver.save(sess, path + '/model-' + str(i) + '.ckpt')
            print("Saved Model")
            
        # Periodically print episodes, total steps conducted, average recent episode 
        # reward, and random action probability
        if len(rList) % 10 == 0:
            print('')
            print("Total Episodes: ", i)
            print("Total Steps: ", total_steps)
            print("Recent Average Reward: ", np.mean(rList[-10:]))
            print("Random Action Probability: ", e)
            print('')
            
    saver.save(sess, path + '/model-' + str(i) + '.ckpt')    # Save final model

###########################
# Checking network learning
###########################
# Look at mean reward over time

# Take the average reward of 10-episode chunks and plot
rMat = np.resize(np.array(rList), [len(rList) // 100, 100])
rMean = np.average(rMat, 1)
plt.plot(rMean)