# Simple Reinforcement Learning with Tensorflow: Part 3 - Model-Based RL

# In this iPython notebook we implement a policy and model network which 
# work in tandem to solve the CartPole reinforcement learning problem. 
# To learn more, read here: https://medium.com/p/9a6fe0cce99

# For more reinforcment learning tutorials, 
# see: https://github.com/awjuliani/DeepRL-Agents

# For info on the CartPole environment (including explanations of the 
# environment observations and action space), see:
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

# For CartPole environment info for the specific version used in this script, see: 
# https://github.com/openai/gym/wiki/CartPole-v0

################
# LOAD LIBRARIES
################
# Done

from __future__ import print_function
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt

import sys
if sys.version_info.major > 2:
    xrange = range
del sys

import gym

############################
# START CARTPOLE ENVIRONMENT
############################
# Done
env = gym.make('CartPole-v0')

##########################
# SETTING HYPER-PARAMETERS
##########################
# Done

H = 8                    # number of hidden layer neurons
learning_rate = 1e-2
gamma = 0.99             # discount factor for reward

# This isn't used in the rest of the script! Not 100% sure what it is...
#decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2

# Not used again in script...
#resume = False # resume from previous checkpoint?

model_bs = 3                # Batch size when learning from model
real_bs = 3                 # Batch size when learning from real environment

# Not used again in script...
# model initialization
#D = 4 # input dimensionality

################
# POLICY NETWORK
################
# Done

tf.reset_default_graph()
observations = tf.placeholder(tf.float32, 
                              [None, 4] , 
                              name = "input_x")

W1 = tf.get_variable("W1", 
                     shape = [4, H], 
                     initializer = tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, W1))

W2 = tf.get_variable("W2", 
                     shape = [H, 1],
                     initializer = tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1, W2)

# There are 2 possible actions. I think this is the probability of action 1.
# See https://github.com/awjuliani/DeepRL-Agents/blob/master/Model-Network.ipynb
probability = tf.nn.sigmoid(score)                

tvars = tf.trainable_variables()

input_y = tf.placeholder(tf.float32,           # These will be reversed actions
                         [None, 1], 
                         name = "input_y")

# These will be scaled and discounted rewards (not necessarily positive)
advantages = tf.placeholder(tf.float32, name = "reward_signal")

adam = tf.train.AdamOptimizer(learning_rate = learning_rate)
W1Grad = tf.placeholder(tf.float32, name = "batch_grad1")
W2Grad = tf.placeholder(tf.float32, name = "batch_grad2")
batchGrad = [W1Grad, W2Grad]

# "input_y" will be reversed actions (when action is 1, "input_y" is 0).
# "action" are governed by "probability". "probability" comes from policy network (take in action, return probability).
# "input_y" is 0 or 1, so 1 of terms of "loglik" goes to 0.
# If "input_y" is 0 ("action" is 1), 1st term is 0 & 2nd term (& whole expression) equals "log(probability)".
# If "input_y" is 1 ("action" is 0), 2nd term is 0 & 1st term (& whole expression) equals "log(1 - probability)".
# This term is the likelihood of "action" given "probability".
loglik = tf.log(input_y * (input_y - probability) + (1 - input_y) * (input_y + probability))

# "advantages" are scaled and discounted rewards & can be negative. "loglik" is always negative.
# If "advantages" is positive, reward is good & "loss" is positive. Minimizing "loss" means maximizing "loglik" which means getting "probability" close to "action".
# If "advantages" is negative, reward is bad & "loss" is negative. Minimizing "loss" means minimizing "loglik" which means getting "probability" far from "action".
# For good "advantages", the network moves "probability" close to the associated "action".
# For bad "advantages", the network moves "probability" away from the associated "action".
loss = -tf.reduce_mean(loglik * advantages)    # "reduce_mean" only computes mean across dimensions (doesn't imply minimization)

newGrads = tf.gradients(loss, tvars)                        # Compute gradients
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))     # Apply gradients

###############
# MODEL NETWORK
###############
# Done
# Implement multi-layer neural network that predicts the next 
# observation, reward, and done state from a current state and action.

mH = 256                      # model layer size

# input_data = tf.placeholder(tf.float32, [None, 5])    # "input_data" doesn't show up again in script

# Don't get what the point of "variable_scope" is...
# 'rnnlm' doesn't show up again in the script.
#with tf.variable_scope('rnnlm'):
#    softmax_w = tf.get_variable("softmax_w", [mH, 50])    # 'softmax_w' doesn't show up again in script
#    softmax_b = tf.get_variable("softmax_b", [50])        # 'softmax_b' doesn't show up again in script

# If the state takes 4 variables to describe in the CartPole environment, 
# state and action (the model inputs) would be a 5-D vector.
previous_state = tf.placeholder(tf.float32, [None, 5] , 
                                name = "previous_state")

W1M = tf.get_variable("W1M", shape = [5, mH],
           initializer = tf.contrib.layers.xavier_initializer())
B1M = tf.Variable(tf.zeros([mH]), name = "B1M")
layer1M = tf.nn.relu(tf.matmul(previous_state, W1M) + B1M)
    
W2M = tf.get_variable("W2M", shape = [mH, mH],
           initializer = tf.contrib.layers.xavier_initializer())
B2M = tf.Variable(tf.zeros([mH]), name = "B2M")
layer2M = tf.nn.relu(tf.matmul(layer1M, W2M) + B2M)

wO = tf.get_variable("wO", shape = [mH, 4],
           initializer = tf.contrib.layers.xavier_initializer())
wR = tf.get_variable("wR", shape = [mH, 1],
           initializer = tf.contrib.layers.xavier_initializer())
wD = tf.get_variable("wD", shape = [mH, 1],
           initializer = tf.contrib.layers.xavier_initializer())

bO = tf.Variable(tf.zeros([4]), name = "bO")
bR = tf.Variable(tf.zeros([1]), name = "bR")
bD = tf.Variable(tf.ones([1]), name = "bD")

# Predictions
predicted_observation = tf.matmul(layer2M, wO, name = "predicted_observation") + bO
predicted_reward = tf.matmul(layer2M, wR, name = "predicted_reward") + bR
predicted_done = tf.sigmoid(tf.matmul(layer2M, wD, name = "predicted_done") + bD)

# Actual outcomes
true_observation = tf.placeholder(tf.float32, [None, 4], name = "true_observation")
true_reward = tf.placeholder(tf.float32, [None, 1], name = "true_reward")
true_done = tf.placeholder(tf.float32, [None, 1], name = "true_done")

predicted_state = tf.concat([predicted_observation, predicted_reward, predicted_done], 1)

# Losses
observation_loss = tf.square(true_observation - predicted_observation)
reward_loss = tf.square(true_reward - predicted_reward)

# "true_done" is either 0 or 1. So one of the terms in "done_loss" always 
# goes to 0. Work out a few examples w/ actual numbers if this doesn't 
# make sense.
done_loss = tf.multiply(predicted_done, true_done) + tf.multiply(1 - predicted_done, 1 - true_done)
done_loss = -tf.log(done_loss)

# "observation_loss" has dimensions "x by 4".
# "done_loss" and "reward_loss" are added as scalars, so "model_loss" also 
# has dimensions "x by 4".
model_loss = tf.reduce_mean(observation_loss + done_loss + reward_loss)

modelAdam = tf.train.AdamOptimizer(learning_rate = learning_rate)
updateModel = modelAdam.minimize(model_loss)

##################
# HELPER FUNCTIONS
##################
# Done

# Set all elements of "gradBuffer" to 0.
def resetGradBuffer(gradBuffer):
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
    return(gradBuffer)
    
# Takes a series of rewards and returns the discounted 
# total reward at each step from that step's point-of-view.
def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return(discounted_r)
    
# Use model to produce a new state when given a previous state and action
def stepModel(sess, xs, action):
    
    # "xs" are previous states. Take most recent state and add the "action".
    toFeed = np.reshape(np.hstack([xs[-1][0], np.array(action)]), 
                        [1, 5])
    
    # Get "predicted_state" (from model network) using "toFeed" as "previous_state".
    myPredict = sess.run([predicted_state], feed_dict = {previous_state: toFeed})
    
    reward = myPredict[0][:, 4]
    observation = myPredict[0][:, 0:4]
    
    # Clip values to within limits that are valid in the Cartpole environment.
    # See: https://github.com/openai/gym/wiki/CartPole-v0
    observation[:, 0] = np.clip(observation[:, 0], -2.4, 2.4)
    observation[:, 2] = np.clip(observation[:, 2], -0.4, 0.4)
    doneP = np.clip(myPredict[0][:, 5], 0, 1)
    
    if doneP > 0.1 or len(xs) >= 300:
        done = True
    else:
        done = False
        
    return(observation, reward, done)

###############################
# TRAINING THE POLICY AND MODEL
###############################
# Done
    
# "xs" / "drs" / "ys" / "ds" are environment states / rewards / (reverse) actions / done statuses
xs, drs, ys, ds = [], [], [], []            
running_reward = None                    # This will be a very slowly changing "reward_sum"
reward_sum = 0                           # Sums reward between switches (over multiple episodes)
episode_number = 1
real_episodes = 1                        # Episodes drawn from real environment (as opposed to the model)
init = tf.global_variables_initializer()
batch_size = real_bs

drawFromModel = False       # When set to True, will use model for observations
trainTheModel = True                     # Whether to train the model
trainThePolicy = False                   # Whether to train the policy

# This governs when:
    # policy network weights update
    # "running_reward" updates
    # update messages print
    # script switches from training model (from real environment) to training policy (from model)
switch_point = 1             

with tf.Session() as sess:                                       # Launch graph
    
    rendering = False          # Don't render environment graphically initially     
    sess.run(init)
    
    observation = env.reset()               # Observation from true environment
    x = observation

    gradBuffer = sess.run(tvars)           # policy network variables (weights)
    gradBuffer = resetGradBuffer(gradBuffer)
    
    # General steps:
    # Run the episode, drawing from environment or model as specified above.
    # When episode ends:
        # increment "episode_number" and maybe "real_episodes"
        # if training model, use episode observations to update model network weights
        # if training policy, use episode observations to compute BUT NOT YET apply policy network weights
        # if switch point condition is met
            # update switch point
            # update policy network weights if training policy
            # update "running_reward"
            # print status stuff
            # switch between training policy from model and model from real environment
            
    # "action" always comes from policy network, even when used in model network. 
    # So as policy network improves, model network episodes last longer.
    
    #while episode_number <= 5000: # ORIGINAL
    while episode_number <= 3000:
        
        # Display environment once performance is acceptably high.
        # "rendering" is only set "True" once performance gets high enough.
        # Not sure why these specific conditions were picked...
        # Rendering really slows down the script...
        if (reward_sum / batch_size > 150 and drawFromModel == False) or rendering == True:
            if real_episodes % 2 == 0:  # I ADDED THIS          
                env.render()             # ORIGINAL
                rendering = True         # ORIGINAL
                #rendering = False
            
        x = np.reshape(observation, [1, 4])                      # Change shape
        
        tfprob = sess.run(probability, feed_dict = {observations: x})  # "probability" is from policy network
        
        action = 1 if np.random.uniform() < tfprob else 0  # Use "tfprob" to probabilistically get an "action"
        
        xs.append(x)   # record various intermediates (needed later for backprop)
        
        # These are reversed here and reversed back later. Not sure why...
        y = 1 if action == 0 else 0              
        ys.append(y)
        
        # step the model or real environment and get new measurements
        
        if drawFromModel == False:  # Draw from real environment using "action" chosen above
            observation, reward, done, info = env.step(action)

        else:                                     # Draw from environment model
            observation, reward, done = stepModel(sess, xs, action)
            
        reward_sum += reward                   # Add reward

        # Add done status. From "env.step()", "done" is "True"/"False", so multiplying by 1 ensures it's 0/1.
        ds.append(done * 1) 
        drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
                    
        if done:   # Pretty much rest of script is under this
            
            if drawFromModel == False:    # Increment "real_episodes" if drawing from real environment
                real_episodes += 1
                
            episode_number += 1                 

            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            # Not 100% sure hidden states and action gradients are above, but 
            # I assume they are done statuses and actions respectively.
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            epd = np.vstack(ds)
            xs, drs, ys, ds = [], [], [], []               # reset array memory  

            # Use episode states/actions to predict next state/rewards/done status.
            # Compare predictions to actual state/reward/done status to update model network weights.
            if trainTheModel == True:
                
               # "y" was set as the flipped version of "action" above. 
               # Now "actions" is set as flipped "y". Not sure why...
               actions = np.array([np.abs(y - 1) for y in epy][:-1])  # All (flipped) actions but last one
               
               state_prevs = epx[:-1,:]               # All states but last one
               state_prevs = np.hstack([state_prevs, actions])
               
               state_nexts = epx[1:, :]             # All states except 1st one
               rewards = np.array(epr[1:, :])      # All rewards except 1st one
               dones = np.array(epd[1:, :])  # All done statuses except 1st one
               
               state_nextsAll = np.hstack([state_nexts, rewards, dones])
               
               feed_dict = {previous_state: state_prevs, 
                            true_observation: state_nexts, 
                            true_done: dones, 
                            true_reward: rewards}
               
               # Update model network weights
               loss, pState, _ = sess.run([model_loss, predicted_state, updateModel], 
                                          feed_dict)
               
            if trainThePolicy == True:
                
                discounted_epr = discount_rewards(epr).astype('float32') # Discount rewards
                
                discounted_epr -= np.mean(discounted_epr)       # Scale rewards
                discounted_epr /= np.std(discounted_epr)
                
                # Compute BUT DO NOT APPLY gradients. "newGrads" is from policy network.
                tGrad = sess.run(newGrads, 
                                 feed_dict = {observations: epx, 
                                              input_y: epy, 
                                              advantages: discounted_epr})
                   
                # If gradients become too large, end training process (ORIGINAL COMMENT)
                # Don't get this, seems like "tGrad[0] == tGrad[0]" will always be "True", 
                # and so the sum will always be >0.
                if np.sum(tGrad[0] == tGrad[0]) == 0:
                    break
                
                # Update "gradBuffer". This does NOT update policy weights.
                # Compute gradients of variables wrt loss based on episode observations.
                # Add gradients to "gradBuffer".
                for ix, grad in enumerate(tGrad):
                    gradBuffer[ix] += grad
                    
            # Initial "switch_point": 1, "batch_size": 3
            # So the 1st time this condition is triggered, is when "episode_number" = 4.
            if switch_point + batch_size == episode_number:  # If switch point condition is met...
                
                switch_point = episode_number             # Update switch point
                
                # If training policy, update policy network weights.
                # So policy network weights only updated every 3 episodes at most.
                if trainThePolicy == True:
                    sess.run(updateGrads, feed_dict = {W1Grad: gradBuffer[0],      
                                                       W2Grad: gradBuffer[1]})
                    gradBuffer = resetGradBuffer(gradBuffer)
                    
                # This is a very slowly changing "reward_sum".
                # Not sure what the point isk. Maybe like a longer term average of performance.
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                
                # If training the model (drawing from real environment)...
                #if drawFromModel == False: # ORIGINAL
                if drawFromModel == False:
                    
                    if real_episodes % 4 == 0:  # I ADDED THIS
                    
                        # Only triggered when episode is done and switch point condition is met.
                        # Print real episodes so far, total batch reward, last action of episode, running batch reward.
                        # The model gets actions from the policy network.
                        # So the reward that the model achieves reflects the performance of the policy network.
                        print('World Perf: Episode %f. Reward %f. action: %f. mean reward %f.' % (real_episodes, 
                                                                                                  reward_sum / real_bs, 
                                                                                                  action, 
                                                                                                  running_reward / real_bs))
                    
                    if reward_sum / batch_size > 200: # If performance is high enough...
                        break
                    
                reward_sum = 0                                   # Reset reward
                
                # Once model has been trained on 100 episodes, start alternating (every batch) between 
                # training policy from model & training model from real environment.
                # This is under the switch point condition.
                if episode_number > 100:
                    drawFromModel = not drawFromModel
                    trainTheModel = not trainTheModel
                    trainThePolicy = not trainThePolicy
                    
            if drawFromModel == True:
                
                # Don't use "env.reset()" b/c we're not drawing from the real environment.
                observation = np.random.uniform(-0.1, 0.1, [4]) # Generate reasonable starting point
                batch_size = model_bs
                
            else:         # Drawing from real environment, so use "env.reset()"
                observation = env.reset()
                batch_size = real_bs
                
print("Real Episodes: ", real_episodes)

###############################
# CHECKING MODEL REPRESENTATION
###############################
# Done
# Here we can examine how well the model is able to approximate the true 
# environment after training. The green line indicates the real environment, 
# and the blue indicates model predictions.

# The plots below are for the last episode. If everything is well-trained, the 
# last episode usually includes ~200 steps.

plt.figure(figsize = (8, 12))

# Use "range(6)" because predictions are defined by 6 values:
    # 4 values define the state
    # 1 value each for the reward and done status
for i in range(6):
    
    plt.subplot(6, 1, i + 1)
    plt.plot(pState[:, i], 'b') # one column from predicted states (from model network)
                                 
    plt.subplot(6, 1, i + 1)
    plt.plot(state_nextsAll[:, i], 'g')    # one column from actual next states    