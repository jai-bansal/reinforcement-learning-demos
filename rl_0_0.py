import gym
import numpy as np

# Create environment.
env = gym.make('FrozenLake-v0')

#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])

# Set learning parameters
lr = .8                            # learning rate
y = .95                            # discount factor
num_episodes = 2000

#create lists to contain total rewards and steps per episode
jList = []
rList = []

for i in range(num_episodes):
    
    print('Episode: ' + str(i))
    
    #Reset environment and get first new observation
    s = env.reset()
    
    rAll = 0
    d = False               # Is the game done or not?
    j = 0
    
    #The Q-Table learning algorithm
    while j < 99:
        
        #print(j)
        j+=1
        
        #Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n) * (1./(i+1)))
        
        #Get new state, reward, done status, and diagnostic info from environment
        s1,r,d,_ = env.step(a)
        
        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr * (r + y * np.max(Q[s1,:]) - Q[s,a])
        
        rAll += r
        s = s1
        
        # Break the loop if the game is over.
        if d == True:
            print('Done')
            break
    print('Steps: ', j)
    print('')
    jList.append(j)
    rList.append(rAll)

print('')
print("Score over time: " +  str(sum(rList)/num_episodes))

print('')
print("Final Q-Table Values")
print(Q)