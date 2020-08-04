# Simple Reinforcement Learning in Tensorflow Part 2-b:

# Vanilla Policy Gradient Agent

# This tutorial contains a simple example of how to build a 
# policy-gradient based agent that can solve the CartPole problem. 
# For more information, see this Medium post:
# https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-2-ded33892c724
# This implementation is generalizable to more than two actions.

# For more Reinforcement Learning algorithms, including DQN 
# and Model-based learning in Tensorflow, see my Github repo, DeepRL-Agents.

# Info on the CartPole environment.
# https://github.com/openai/gym/wiki/CartPole-v0

# IMPORT MODULES
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as plt
#%matplotlib inline

# Make script compatible with Python 2 and 3?
try:
    xrange = xrange
except:
    xrange = range
    
# CREATE ENVIRONMENT
env = gym.make('CartPole-v0')

# THE POLICY-BASED AGENT

gamma = 0.99                         # discount rate

# Define function that takes a series of rewards and returns the discounted 
# total reward at each step from that step's point-of-view.
def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return(discounted_r)
    
# Define agent class
class agent():
    
    # "s_size": size of state space (how many values are needed to describe the state)
    # "a_size": action space size?
    # "h_size": # of nodes in hidden layer
    def __init__(self, lr, s_size, a_size, h_size):
    
        # These lines establish the feed-forward part of the network. 
        # The agent takes a state and produces an action.
        
        self.state_in = tf.placeholder(shape = [None, s_size], dtype = tf.float32)
        
        hidden = slim.fully_connected(self.state_in, 
                                      h_size, 
                                      biases_initializer = None, 
                                      activation_fn = tf.nn.relu)
        self.output = slim.fully_connected(hidden, 
                                           a_size, 
                                           activation_fn = tf.nn.softmax, 
                                           biases_initializer = None)
        self.chosen_action = tf.argmax(self.output, 1)
        
        # The next lines establish the training procedure. 
        # We feed the reward and chosen action into the network 
        # to compute the loss, and use it to update the network.
        
        self.reward_holder = tf.placeholder(shape = [None], dtype = tf.float32)
        self.action_holder = tf.placeholder(shape = [None], dtype = tf.int32)    
        
        # This took me a while to get. Suppose "output" consists of 3 timesteps.
        # And the actions for those timesteps are (0, 1, 0) (binary actions).
        # The "range" timesteps are (0, 1, 2) and the multiplication below yields (0, 2 ,4).
        # After addition of the actions, I get (0, 3, 4).
        # If "outputs" is made 1-D (as in the next line down), (0, 3, 4) are the 
        # indices I want that are "responsible" for the observed rewards.
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
    
        # Get values for the relevant actions.
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), 
                                             self.indexes)
        
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)
        
        tvars = tf.trainable_variables()                  # Get all trainable variables.
        self.gradient_holders = []
        
        # Create named placeholders for all trainable variables.
        # Add the named placeholder to "self.gradient_holders".
        # They're replaced by actual gradients ("gradBuffer") later.
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name = str(idx) + '_holder')
            self.gradient_holders.append(placeholder)       
                                                         
        self.gradients = tf.gradients(self.loss, tvars)         # Compute gradients
        
        optimizer = tf.train.AdamOptimizer(learning_rate = lr)  # Create optimizer
        
        # When "update_batch" is called, actual gradients are supplied.
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))
        
# TRAINING THE AGENT
        
tf.reset_default_graph() # Clear the Tensorflow graph.

myAgent = agent(lr = 1e-2, s_size = 4, a_size = 2, h_size = 8) # Load the agent.

total_episodes = 5000     # Set total number of episodes to train agent on.
max_ep = 999              # Seems to be maximum episode length.
update_frequency = 5      # Set how often to update weights.

init = tf.global_variables_initializer()

# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    i = 0
    total_reward = []
    total_length = []               # Will hold length of each episode.
        
    # This takes the trainable variables, puts them in "gradBuffer", and sets them to 0. They'll be updated later.
    gradBuffer = sess.run(tf.trainable_variables())
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
        
    while i < total_episodes:
        
        s = env.reset()
        running_reward = 0
        ep_history = []
        
        for j in range(max_ep):
            
            # Probabilistically pick an action given our network outputs.
            a_dist = sess.run(myAgent.output,                       # "a_dist" stands for action distribution?
                              feed_dict = {myAgent.state_in : [s]})
            a = np.random.choice(a_dist[0], p = a_dist[0])
            
            # "a" contains probabilities. This line makes "a" contain an action index.
            a = np.argmax(a_dist == a)
            
            s1, r, d, _ = env.step(a) #Get our reward for taking an action ("d" is done status of the game).
            ep_history.append([s, a, r, s1])
            s = s1
            running_reward += r
            
            # If game is done...
            if d == True:
                
                #Update network.
                ep_history = np.array(ep_history)
                ep_history[:, 2] = discount_rewards(ep_history[:, 2])
                
                feed_dict = {myAgent.reward_holder : ep_history[:, 2], 
                             myAgent.action_holder : ep_history[:, 1], 
                             myAgent.state_in : np.vstack(ep_history[:, 0])}
                
                grads = sess.run(myAgent.gradients, feed_dict = feed_dict)
                
                # My exploration code.
                #grads, ind, o, sh, a, b, c, d, e, ro = sess.run([myAgent.gradients, myAgent.indexes, myAgent.output, tf.shape(myAgent.output), 
                #                                    tf.shape(myAgent.output)[0], tf.shape(myAgent.output)[1], tf.range(0, tf.shape(myAgent.output)[0]),                                                 
                #                                    tf.range(0, tf.shape(myAgent.output)[0]) * tf.shape(myAgent.output)[1], 
                #                                   tf.range(0, tf.shape(myAgent.output)[0]) * tf.shape(myAgent.output)[1] + myAgent.action_holder, 
                #                                    myAgent.responsible_outputs], 
                #                              feed_dict = feed_dict)
                
                # Update gradients.
                for idx, grad in enumerate(grads):
                    gradBuffer[idx] += grad              # Becomes a kind of cumulative gradient.
                    
                # Every so often, update weights.
                # Note that the game must be over AND "i % update_frequency == 0" 
                # for the following to trigger.
                if i % update_frequency == 0 and i != 0:
                    
                    feed_dict = dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                    
                    _ = sess.run(myAgent.update_batch, feed_dict = feed_dict)   # Update weights
                    
                    # Reset "gradBuffer".
                    for ix, grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0
                        
                # Add reward and game length for this episode.
                total_reward.append(running_reward)
                total_length.append(j)
                break
            
        # Every so often, print mean reward.
        if i % 100 == 0:
            print(np.mean(total_reward[-100:]))
            
        i += 1          # Increment episode.               
                    