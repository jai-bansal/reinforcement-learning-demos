# IMPORT MODULES
import  gym
import  numpy  as  np
import  random
import  tensorflow as tf
import matplotlib.pyplot as plt
#%matplotlib inline

# Load environment.
env = gym.make('FrozenLake-v0')

# Q-NETWORK APPROACH

# Implementing the network itself

# Reset graph.
tf.reset_default_graph()

# Build network.
inputs1 = tf.placeholder(shape=[1,16],                 # Specify placeholder for network inputs.
                         dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16,4], 0, 0.01))    # Specify weights.
Qout = tf.matmul(inputs1, W)                           # Get output Q-values.
predict = tf.argmax(Qout, 1)                           # Choose action (index of maximum "Qout").

#Below we obtain the loss by taking the sum of squares difference 
# between the target and prediction Q values.

# Specify placeholder for target Q-values.
nextQ = tf.placeholder(shape=[1,4], 
                       dtype=tf.float32)

# Define loss (sum of squares between predicted and target Q-values).
loss = tf.reduce_sum(tf.square(nextQ - Qout))

# Define trainer.
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

# Minimize loss using trainer.
updateModel = trainer.minimize(loss)

# TRAINING THE NETWORK

# Initialize variables.
init = tf.global_variables_initializer()

# Set learning parameters
y = .99                                 # discount rate
e = 0.1                                 # paramater that governs how often a random action is chosen
num_episodes = 2000

#create lists to contain total rewards and steps per episode
jList = []
rList = []

# Run it.
with tf.Session() as sess:
    sess.run(init)
    
    # Loop through episodes.
    for i in range(num_episodes):
        
        
        #Reset environment and get first new observation
        s = env.reset()
        
        # Set initial parameters.
        rAll = 0                            # Total reward starts at 0
        d = False                           # Initial state is not done
        j = 0                               # Current step is 0
        
        #The Q-Network
        while j < 99:
            
            # Increment step.
            j+=1
        
            # Choose an action by greedily (with e chance of random action) 
            # from the Q-network
            
            # Get Q-value and predicted action for initial state.
            a, allQ = sess.run([predict, Qout], 
                               feed_dict = {inputs1 : np.identity(16)[s : s+1]})
            
            # Sometimes, randomly select an action (instead of the best action
            # according to Q-value).
            if np.random.rand(1) < e:
                    a[0] = env.action_space.sample()
                    
            # Get new state, reward, done status, 
            # and diagnostic info from environment.
            s1, r, d, _ = env.step(a[0])
            
            # Obtain Q-values for new state.
            Q1 = sess.run(Qout, 
                          feed_dict = {inputs1 : np.identity(16)[s1 : (s1 + 1)]})
            
            # Get max action value for new state.
            maxQ1 = np.max(Q1)
            
            # Get target value for chosen state/action pair.
            # This is "allQ", but with the relevant action's Q-value modified.
            # How? With the reward of that action plus discounted value of the 
            # best action in the next state (r + y * maxQ1).
            targetQ = allQ
            targetQ[0,a[0]] = r + (y * maxQ1)
            
            # Train network using target and predicted Q values.
            _, W1 = sess.run([updateModel, W], 
                             feed_dict = {inputs1 : np.identity(16)[s : s+1], 
                                          nextQ : targetQ})
            
            # Add any reward obtained to "rAll".
            rAll += r
            
            # Update "s".
            s = s1
            
            # If the action results in the game being done...
            if d == True:
                
                    #Reduce chance of random action as we train the model 
                    # and break out of the loop.
                    e = 1. / ((i/50) + 10)
                    break
                
        # At the end of the episode, add the # of steps and total reward.
        jList.append(j)
        rList.append(rAll)
        
print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")

# SOME STATS ON NETWORK PERFORMANCE

plt.plot(rList)
plt.plot(jList)
  
        
        
        
        

