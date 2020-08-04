#######################################################
# SIMPLE REINFORCEMENT LEARNING: EXPLORATION STRATEGIES
#######################################################
# This notebook contains implementations of various action-selections methods 
# that can be used to encourage exploration during the learning process. 
# To learn more about these methods, see this Medium post: 
# https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-7-action-selection-strategies-for-exploration-d3a97b7cceaf
 
# Also see the interactive visualization: http://awjuliani.github.io/exploration/index.html

# For more reinforcment learning tutorials see: https://github.com/awjuliani/DeepRL-Agents

# For info on the CartPole environment (including explanations of the 
# environment observations and action space), see:
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

# For CartPole environment info for the specific version used in this script, see: 
# https://github.com/openai/gym/wiki/CartPole-v0

##################
# IMPORT LIBRARIES
##################
from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
#%matplotlib inline

import tensorflow.contrib.slim as slim

##################
# LOAD ENVIRONMENT
##################
env = gym.make('CartPole-v0')      # Create instance of Cartpole v0 environment

##################
# HELPER FUNCTIONS
##################

# Create class to store experiences and sample them randomly for network training
class experience_buffer():
    def __init__(self, buffer_size = 10000):
        
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self, experience):   # Function to add new "experience" to "buffer"
        
        # If adding new "experience" to "buffer" makes "buffer" too big (larger than "buffer_size"), 
        # remove experiences from the front of "buffer" to make room for new experiences.
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0: (len(experience) + len(self.buffer)) - self.buffer_size] = []
            
        self.buffer.extend(experience)
    
    # Sample experiences (consisting of original state, action, reward, new state, and done status)
    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])
    
# These functions slowly update target network parameters to primary network parameters.
# Idea: having a target network without constantly shifting Q-values makes training more stable
        
# Below, variables are structured so the 1st half are from the primary network and 
# the 2nd half are from target network.
# "tau" is the rate to update target network toward primary network.
# Target network parameters slowly move towards primary network parameters.
    
def updateTargetGraph(tfVars, tau):
    
    total_vars = len(tfVars)
    op_holder = []
    
    for idx, var in enumerate(tfVars[0: total_vars // 2]): # Loop through 1st half of variables
        
        # Update 2nd half of variables. Assign variable as a combo (based on "tau") of "var" and original variable value.
        op_holder.append(tfVars[idx + total_vars // 2].assign((var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
        
    return op_holder

# This is unintuitive structure to me, but is used below to run "updateTargetGraph". 
# "updateTargetGraph" is used to move target network parameters towards primary network parameters.
def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)
        
#############################
# IMPLEMENTING DEEP Q-NETWORK
#############################
        
class Q_Network():
    def __init__(self):
        
        # These lines establish the feed-forward part of the network used to choose actions
        
        self.inputs = tf.placeholder(shape=[None, 4], dtype = tf.float32) # CartPole environment observations are defined by 4 values
        self.Temp = tf.placeholder(shape = None, dtype = tf.float32) # Parameter used in Boltzmann exploration approach
        self.keep_per = tf.placeholder(shape = None, dtype = tf.float32) # Used for dropout

        hidden = slim.fully_connected(self.inputs, 64,                                         # Create fully connected layer
                                      activation_fn = tf.nn.tanh, biases_initializer = None)
        hidden = slim.dropout(hidden, self.keep_per)                  # Dropout

        # Another fully connected layer
        self.Q_out = slim.fully_connected(hidden, 2,                                        # CartPole environment has 2 possible actions
                                          activation_fn = None, biases_initializer = None)

        self.predict = tf.argmax(self.Q_out, 1)                   # Pick action
        
        # Generate "probabilities" for Boltzmann exploration approach using "self.Temp" parameter
        self.Q_dist = tf.nn.softmax(self.Q_out / self.Temp)

        # Get loss by taking the sum of squares difference between target and prediction Q values
        
        self.actions = tf.placeholder(shape = [None], dtype = tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, 2, dtype = tf.float32)

        # "self.Q_out" are predicted action Q-values for each input state.
        # "self.actions_onehot" are the one-hot version of chosen actions.
        # So, "tf.multiply(self.Q_out, self.actions_onehot)" are the predicted Q-values for chosen actions.
        # "tf.reduce_sum" removes 0s.
        self.Q = tf.reduce_sum(tf.multiply(self.Q_out, self.actions_onehot), 
                               axis = 1)

        self.nextQ = tf.placeholder(shape = [None], dtype = tf.float32) # Target Q-values
        loss = tf.reduce_sum(tf.square(self.nextQ - self.Q))     # Compute loss
        trainer = tf.train.GradientDescentOptimizer(learning_rate = 0.0005)
        self.updateModel = trainer.minimize(loss)
        
############
# PARAMETERS
############

exploration = "greedy" # Exploration method. Choose between: greedy, random, e-greedy, boltzmann, bayesian.
y = 0.99                         # Discount factor for Q-values of "next" state
num_episodes = 20000            # Total number of episodes to train network for
tau = 0.001 # Rate to update target network parameters toward primary network parameters
batch_size = 32                                           # Training batch size
startE = 1                                  # Initial random action probability [ORIGINAL CODE]
endE = 0.1                                    # Final random action probability [ORIGINAL CODE]
#startE = 10                                                           # MY TEST
#endE = 0.75                                                              # MY TEST
annealing_steps = 200000         # Number of steps to reduce "startE" to "endE"
pre_train_steps = 50000                # Number of steps before training begins

###############        
# TRAIN NETWORK
###############

tf.reset_default_graph()                           # Reset global default graph

q_net = Q_Network()                                         # Create Q-networks
target_net = Q_Network()

init = tf.global_variables_initializer() # Returns op that initializes global variables
trainables = tf.trainable_variables()

# Returns operation to update target network parameters towards primary network parameters
targetOps = updateTargetGraph(trainables, tau)

myBuffer = experience_buffer()                       # Create experience buffer

# Create lists to contain total rewards and steps per episode
jList = []                                          # List of steps per episode
jMeans = []        # Save average number of steps over last 100 steps over time
rList = []                                  # List of total rewards per episode
rMeans = []                 # Save average reward over last 100 steps over time

## CODE REVIEW DONE TO HERE

with tf.Session() as sess:
    sess.run(init)

    # This is unintuitive structure to me, but is used to run "updateTargetGraph". 
    # "updateTargetGraph" is used to move target network parameters towards primary network parameters.
    updateTarget(targetOps, sess)
    
    e = startE                                 # Set initial random action rate
    stepDrop = (startE - endE) / annealing_steps # Define random action probability decrease per step
    
    total_steps = 0

    for i in range(num_episodes):
        
        s = env.reset()             # Reset environment and get 1st observation
        rAll = 0                                       # Episode reward tracker
        d = False                                                 # Done status
        j = 0                                            # Episode step tracker

        while j < 999:                        # Episode is limited to 999 steps...remember this is the CartPole environment
            j += 1

            # "greedy" performs the worst...average reward 9.37. Reward doesn't really change over time. Pretty much replicated.
            # I'm surprised it was even worse than "random".
            # Setting "keep_per" = (1 - e) + 0.1 makes this the same as "Bayesian".
            # Performance was similar to "Bayesian" but not exactly the same...interesting to see the variation within the same algo.
            if exploration == "greedy": # Choose action with maximum expected value
                
                a, allQ = sess.run([q_net.predict, q_net.Q_out],      # Get chosen action and action Q-values
                                   feed_dict = {q_net.inputs: [s], 
                                                q_net.keep_per: 1.0}) # No dropout # ORIGINAL CODE
                                                #q_net.keep_per: (1 - e) + 0.1}) # MY TEST
                a = a[0]

            # Average reward 22.38, reward doesn't really change over time. Almost exactly replicatd.
            if exploration == "random":                # Choose action randomly
                a = env.action_space.sample()

            # Average reward 27.65, reward increases over time but then somewhat decreases. Mostly replicated.
            # Setting "keep_per" = 0.85 results in average reward going up to 56.6. This is NOT always replicated.
            # "Bayesian" dropout (setting "keep_per" = (1 - e) + 0.1) does NOT work with "e-greedy".
            if exploration == "e-greedy": # Choose action greedily but with "e" probability of random action
                
                # Pick random action w/ "e" chance. Definitely pick random action for the 1st "pre_train_steps" steps.
                if np.random.rand(1) < e or total_steps < pre_train_steps:
                    a = env.action_space.sample()
                    
                else:                                                     # Same as "greedy" approach
                    a, allQ = sess.run([q_net.predict, q_net.Q_out],      # Get chosen action and action Q-values
                                       feed_dict = {q_net.inputs: [s], 
                                                    q_net.keep_per: 1.0}) # No dropout # ORIGINAL CODE
                                                    #q_net.keep_per: 0.85}) # MY TEST
                                                    #q_net.keep_per: (1 - e) + 0.1}) # MY TEST
                    a = a[0]

            # Average reward 11.0, with reward decreasing drastically over time.
            # This always sucked despite testing a variety of parameters.
            # Also tested "startE" = [2, 5, 7.5, 7.5, 10, 25, 50], "endE" = [0.1, 0.1, 0.25, 0.5, 0.75, 0.5, 0.5], not much improvement (~16 or 17 Overall Average Episode Reward)
            # Setting "keep_per" = 0.85 gets average reward of 26.3, but reward doesn't really increase over time. This isn't always reproducible.
            # Suprisingly bad...
            if exploration == "boltzmann": # Choose action probabilistically, with action probabilities proportionate to Q-values
                
                # Temperature controls the spread of the softmax distribution, 
                # such that all actions are considered relatively equally at the start of training, 
                # and actions are sparsely distributed by the end of training.
                
                Q_d, allQ = sess.run([q_net.Q_dist, q_net.Q_out],      # Get modified and original action Q-values
                                    feed_dict = {q_net.inputs: [s], 
                                                 q_net.Temp: e,        # "e" drops from 1 to 0.1 over time
                                                 q_net.keep_per: 1.0}) # No dropout
                                                 #q_net.keep_per: 0.85}) # My test
    
                a = np.random.choice(Q_d[0], p = Q_d[0]) # Pick from values of "Q_d" using "Q_d" values as relevant probabilities
                a = np.argmax(Q_d[0] == a)

            # Choose action using a sample from a dropout approximation of a Bayesian Q-network [ORIGINAL COMMENT]
            # Not sure I understand the above comment...
            # I don't understand how taking a SINGLE sample from a network w/ dropout gives any measure of uncertainty.
            # It's just a noisy estimate (because of the dropout). The "noise" does decrease as "keep_per" increases from 0.1 to 1 as stated in the blog.
            # Not sure how agent is exploiting its uncertainty about its actions...
            # I don't really get the "Shortcoming" section of the relevant part of the blog either...
            # If we were taking multiple samples, that would make sense to me and provide a measure of uncertainty that could be used.
            # Below, it seems we're just running the "greedy" approach w/ an increasing "keep_per" rate.
            # Average reward: 88.91 (generally replicated)...I interpret this as saying dropout is really helpful more than any Bayesian sampling scheme.
            if exploration == "bayesian":
                
                a, allQ = sess.run([q_net.predict, q_net.Q_out],                # Get chosen action and action Q-values
                                   feed_dict = {q_net.inputs: [s], 
                                                q_net.keep_per: (1 - e) + 0.1}) # Some dropout! "keep_per" starts at 0.1, ends at 1. Inititally, drops out a lot, eventually none.
                a = a[0]

            s1, r, d, _ = env.step(a) # Take action "a" and get new state, reward, and done status
            
            myBuffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 5])) # Add experience to buffer
            
            # After first "pre_train_steps", "e" decreases by "stepDrop" until it hits "endE".
            # "e" can be different things depending on which method you're using.
            if e > endE and total_steps > pre_train_steps:
                e -= stepDrop

            if total_steps > pre_train_steps and total_steps % 5 == 0: # After first "pre_train_steps" steps, update model every 5 steps
                
                # Use Double-DQN training algorithm
                
                trainBatch = myBuffer.sample(batch_size)   # Sample experiences

                # Outputs are actions picked by MAIN network using RESULT STATES as input
                Q1 = sess.run(q_net.predict, 
                              feed_dict = {q_net.inputs: np.vstack(trainBatch[:, 3]),  # Inputs are result states!
                                           q_net.keep_per: 1.0})                       # No dropout
    
                # Outputs are Q-values picked by TARGET network using RESULT STATES as input
                Q2 = sess.run(target_net.Q_out, 
                              feed_dict = {target_net.inputs: np.vstack(trainBatch[:, 3]), # Inputs are result states!
                                           target_net.keep_per: 1.0})                      # No dropout
                   
                # 4th index of "trainBatch" is done status: "True" or "False"
                # These correspond to 1 or 0 respectively.
                # Result: "True"/"False" becomes 0/1 (flipped).
                # This will be a multiplier for target Q-values.
                end_multiplier = -(trainBatch[:, 4] - 1)
                
                # "Q2" are vectors of Q-values (1 per action) from TARGET network.
                # "Q1" are action indices chosen by MAIN network.
                # Result: "batch_size" Q-values each corresponding to an action chosen by main network.
                # Recall that these are actions chosen when RESULT STATES are fed in as inputs.
                doubleQ = Q2[range(batch_size), Q1]
                
                # 2nd index of "trainBatch" is reward.
                # So, target Q-value is immediate reward + discounted Q-value looking out from next state/action pair.
                # Because of "end_multiplier", 2nd term is 0 if a move ends the game and 1 otherwise.
                # So, if a move ends the game, 2nd term is 0 b/c there's no Q-value looking out from the done state.
                targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)
                
                _ = sess.run(q_net.updateModel,                                       # Update
                             feed_dict = {q_net.inputs: np.vstack(trainBatch[:, 0]),  # Initial states
                                          q_net.nextQ: targetQ,                       # Target Q-values derived above
                                          q_net.keep_per: 1.0,                        # No dropout
                                          q_net.actions: trainBatch[:, 1]})           # Actions

                # This is unintuitive structure to me, but is used to run "updateTargetGraph". 
                # "updateTargetGraph" is used to move target network parameters towards primary network parameters.
                updateTarget(targetOps, sess)
                
                # End update

            rAll += r                    # Add reward to episode reward tracker
            s = s1                                               # Update state
            total_steps += 1                                  # Increment steps
            
            if d == True:              # If episode is done, break episode loop
                break
 
        jList.append(j)                          # Add episode steps to tracker
        rList.append(rAll)                      # Add episode reward to tracker

        if i % 100 == 0 and i != 0:                            # Report metrics
            
            r_mean = np.mean(rList[-100: ]) # Average reward over last 100 episodes
            j_mean = np.mean(jList[-100: ]) # Average number of steps over last 100 episodes
            
            rMeans.append(r_mean)                                        # Save
            jMeans.append(j_mean)
            
            if i % 1000 == 0:
            
                print('')
                print("Episodes:", i)
                print("Average Recent Reward:", str(r_mean))
                print("Total Steps: ", str(total_steps))
    
                if exploration == 'e-greedy':
                    print("Epsilon:", str(round(e, 2)))
                    
                if exploration == 'boltzmann':
                    print("Temp:", str(round(e, 2)))
                    
                if exploration == 'bayesian':
                    #print("p:", str(round(e, 2)))      # Not sure what's the point of reporting this...
                    print("Keep %:", (1 - e) + 0.1)
                    
print("Exploration Method:", exploration)
print("Overall Average Episode Reward: ", str(sum(rList) / num_episodes))

#################
# ANALYZE RESULTS
#################

#plt.figure(figsize = (14, 10))                                 # Set figure size
#plt.plot(rMeans)                                # Plot episode reward over time

#plt.figure(figsize = (14, 10))                                 # Set figure size
#plt.plot(jMeans)                                 # Plot episode steps over time
        
