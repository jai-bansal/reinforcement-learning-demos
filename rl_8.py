## D1
##### SIMPLE REINFORCEMENT LEARNING WITH TENSORFLOW: PART 8: ASYNCHRONOUS ADVANTAGE ACTOR-CRITIC (A3C)

# This script implements the A3C algorithm to solve a simple 3D Doom challenge using the VizDoom engine.
# For more info, see this Medium post:
# https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2

# This tutorial requires installation of VizDoom. Install with this command: pip install vizdoom
# You also need "basic.wad" and "helper.py". Both are in the relevant repo: https://github.com/awjuliani/DeepRL-Agents

# During training, Tensorboard shows agent performance info. Launch it with:
# tensorboard --logdir=worker_0:'./train_0',worker_1:'./train_1',worker_2:'./train_2',worker_3:'./train_3'

##### IMPORT MODULES

import os
os.chdir("C:/Users/jbans/Desktop/RL")
import threading
import multiprocessing
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from time import sleep, time
from vizdoom import *
from skimage.transform import resize

# Must download repo (https://github.com/awjuliani/DeepRL-Agents) and have "helper.py" file in working directory.
# "helper.py" contains helper functions. Don't fully get "helper.py"...see that file for more info
from helper import *

##### HELPER FUNCTIONS

# Copies 1 set of variables to another. Used to set worker network params to those of global network.
# Below it's used as: update_target_graph('global', self.name)
# Seems like "from_scope" is global network and "to_scope" is a worker.
def update_target_graph(from_scope, to_scope):

    # "tf.get_collection": returns list of values in the collection with the given name
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):

        # 'op_holder' holds (I assume) the updated variables. Not sure why 'op_holder' is needed...
        # I assume variables are updated as that's the point, but not totally sure...
        op_holder.append(to_var.assign(from_var))
    return(op_holder)

def process_frame(frame):          # Crop and resize original Doom screen image
    s = frame[10:-10, 30:-30]                         # Crop image to 100 x 100...not sure why these exact values were used
    # s = scipy.misc.imresize(s, [84, 84])              # Resize...images are black and white, not sure why these values were used... (deprecated?)
    s = resize(s, (84, 84)) # 'imresize' is deprecated, this line is the replacement
    s = np.reshape(s, [np.prod(s.shape)]) / 255.0     # Make 1-D and scale
    return(s)

# Computes total discounted reward for rest of episode from the perspective of each state.
# Run with "x" = "[1, 1, 1]" to understand. "x[::-1]" reverses "x". "gamma": discount rate.
# I don't understand "lfilter", but it does this task...turns out there's a signal processing function that does exactly what's needed
# I would've written a longer but more readable function to do this...
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis = 0)[::-1]

def normalized_columns_initializer(std = 1.0):   # Returns function to initialize weights. Used below for policy and value output layers.
    def _initializer(shape, dtype = None, partition_info = None): # I assume "shape" comes from "fully_connected" command below...

        # Pick values from standard normal distribution. "*" is unpacking operator and allows a tuple argument.
        out = np.random.randn(*shape).astype(np.float32)

        out *= std / np.sqrt(np.square(out).sum(axis = 0, keepdims = True)) # Not sure why this exact weight scaling is used...
        return tf.constant(out)  # Not sure why "tf.constant()" is needed...
    return(_initializer)

##### ACTOR-CRITIC NETWORK
# Create global network. Take frames, preprocess it, run it through conv layers, then LSTM layers,
# then split into value and policy layers. For worker instances, compute losses, gradients, and propagate.

# "s_size": # of pixels (84 * 84 = 7056). "a_size": # of actions (move left, move right, fire).
# "scope": naming convention used to keep different sets of variables separate (differentiates btwn global network and workers).
# "trainer": optimizer (will be "tf.train.AdamOptimizer()").
class AC_Network():
    def __init__(self, s_size, a_size, scope, trainer):

        # "scope": naming convention used to keep different sets of variables separate (global vs. workers).
        with tf.variable_scope(scope):

            # Input and visual encoding layers
            self.inputs = tf.placeholder(shape = [None, s_size], dtype = tf.float32) # "s_size" is 1-D length of a processed frame
            self.imageIn = tf.reshape(self.inputs, shape = [-1, 84, 84, 1])

            self.conv1 = slim.conv2d(activation_fn = tf.nn.elu,  # Convolutional layers
                                     inputs = self.imageIn,
                                     num_outputs = 16,
                                     kernel_size = [8, 8],
                                     stride = [4, 4],
                                     padding = 'VALID')
            self.conv2 = slim.conv2d(activation_fn = tf.nn.elu,
                                     inputs = self.conv1,
                                     num_outputs = 32,
                                     kernel_size = [4, 4],
                                     stride = [2, 2],
                                     padding = 'VALID')

            hidden = slim.fully_connected(slim.flatten(self.conv2),  # "slim.flatten" turns a (?, 9, 9, 32) tensor to a (?, 2592) tensor
                                          256,
                                          activation_fn = tf.nn.elu)

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple = True)  # LSTM for temporal dependencies
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32) # Initialize LSTM output and memory with 0s
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]

            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)

            rnn_in = tf.expand_dims(hidden, [0]) # Add dimension of 1 to front of "hidden", this will be batch size below
            step_size = tf.shape(self.imageIn)[:1] # Get 1st dim of "imageIn" shape (# of frames). This defines how many times the RNN unrolls.

            # Weird construction. Both "self.state_in" and "state_in" are fed by "c_in" and "h_in" placeholders.
            # I'm not totally sure if "self.state_in" or "state_in" are used below, but doesn't matter...
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in) # Puts "c_in" (hidden state) and "h_in" (output) in tuple for LSTM

            # See https://stackoverflow.com/questions/41885519/tensorflow-dynamic-rnn-parameters-meaning
            # Feed "rnn_in" into "lstm_cell". "lstm_outputs" / "lstm_state": RNN output / final state.
            # "time_major = False" means input/output tensors have shape: [batch_size, max_time, input_size]
            # "max_time": # of steps (unrolls) in the longest sequence. Here, there's only 1 sequence, ie. batch_size = 1.
            # "rnn_in" dims: (1, ?, 256). That's [batch_size, max_time, input_size]
            # "lstm_outputs" has dims (batch_size x max_time x output_size) = (1 x ? x 256). I believe it's RNN outputs for each step.
            # "lstm_state" is final state. It has c and h both w/ dims (1 x 256).
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell,
                                                         rnn_in,
                                                         initial_state = state_in,
                                                         sequence_length = step_size,
                                                         time_major = False)
            lstm_c, lstm_h = lstm_state  # Final hidden state and output

            # Dims of "lstm_c[:1, :]" / "lstm_h[:1, :]" are the same as "lstm_c" / "lstm_h" respectively. Not sure what the point is...
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])

            rnn_out = tf.reshape(lstm_outputs, [-1, 256])  # Reshape "lstm_outputs" from (1 x ? x 256) to (? x 256)

            self.policy = slim.fully_connected(rnn_out,     # Output layers for policy and value estimations
                                               a_size,
                                               activation_fn = tf.nn.softmax,
                                               weights_initializer = normalized_columns_initializer(0.01),  # function defined above initializes weights
                                               biases_initializer = None)
            self.value = slim.fully_connected(rnn_out,
                                              1,
                                              activation_fn = None,
                                              weights_initializer = normalized_columns_initializer(1.0),
                                              biases_initializer = None)

            if scope != 'global': # Only workers needs ops for loss functions and gradient updating
                self.actions = tf.placeholder(shape = [None], dtype = tf.int32)               # Actions supplied below
                self.actions_onehot = tf.one_hot(self.actions, a_size, dtype = tf.float32)
                self.target_v = tf.placeholder(shape = [None], dtype = tf.float32)            # Discounted rewards below, used in "self.value_loss"
                self.advantages = tf.placeholder(shape = [None], dtype = tf.float32)          # Defined below. Used in "self.policy_loss"

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])  # Policy weights ONLY for chosen actions

                # Loss functions: high "value_loss" and "policy_loss" are bad, high "entropy" is good (to an extent)
                # Not sure why these exact forms are used...
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))  # Diff btwn discounted rewards and value

                # Measure of spread of action probs. I think "self.policy" are action probs. entropy is positive.
                # If prob = 1, for 1 action, entropy = 0. If prob = ~0.33 for each action, entropy = ~1.1.
                # Higher entropy means probs are spread out across actions more. Entropy used to improve exploration.
                self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))

                # I think "policy" are probs (btwn 0 and 1), so "responsible_outputs" are btwn 0 and 1.
                # So "log(responsible_outputs)" is < 0 and "policy_loss" is > 0 (if advantage is > 0).
                # Want to lower "policy_loss". If "advantages" > 0 [< 0], "policy_loss" is positive [negative].
                # If "advantages" > 0 [< 0] (a good [bad] action), lowering "policy_loss" means increasing [decreasing] "responsible_outputs". 
                # So for good [bad] actions, the probability of doing them increases [decreases]. 
                # How does this pick btwn good and even better actions? Over time, actions get better and state value increases.
                # So actions that used to have positive advantage will eventually have negative advantage. 
                # Until only the best actions have positive advantage.
                # I guess this is how A3C works and A3C has good results...              
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages) # "advantages" defined below

                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)     # Compute 1 normalized value for all "local_vars"
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)  # "grads" are clipped tensors, "grad_norms" is the global norm

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))  # "trainer" is an optimizer

##### WORKER AGENT

# "game": Doom game object. "name": names worker. "s_size": # of pixels (84 * 84 = 7056).
# "a_size": length of action space (move left, move right, fire). Other variables listed below
# "global_episodes" seems to count episodes only for "worker_0" below.
# Not sure why...perhaps it counts how many times all workers run? That assumes all workers run episodes in an equal length of time...
class Worker():
    def __init__(self, game, name, s_size, a_size, trainer, model_path, global_episodes):

        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path                          # "model_path": save path for model objects
        self.trainer = trainer                                # "trainer": an optimizer ("tf.train.AdamOptimizer()" below)
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)   # Add 1 to "global_episodes". It's set to 0 initially below, maybe that's why...
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []                         # holds average episode state values (1 value per episode)
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))  # "summary" provides a way to store summary stats

        self.local_AC = AC_Network(s_size, a_size, self.name, trainer)   # Create local copy of network
        self.update_local_ops = update_target_graph('global', self.name) # Create TF op to copy global params to local network

        # This code relates to setting up Doom environment. "basic.wad" is unintuitive to read but sets up scenario I think.
        # This page has helpful explanations: http://vizdoom.cs.put.edu.pl/tutorial

        # "basic.wad" defines how a world works and looks like (maps)
        game.set_doom_scenario_path("basic.wad") # Corresponds to the simple task we pose our agent

        game.set_doom_map("map01")
        game.set_screen_resolution(ScreenResolution.RES_160X120) # Set screen resolution
        game.set_screen_format(ScreenFormat.GRAY8) # Set screen buffer format...not 100% sure what this means but doesn't seem important, maybe color vs. black/white
        game.set_render_hud(False)        # Sets whether to render particular visual elements
        game.set_render_crosshair(False)
        game.set_render_weapon(True)
        game.set_render_decals(False)     # Not sure what these last 2 are, but doesn't seem important
        game.set_render_particles(False)

        game.add_available_button(Button.MOVE_LEFT)   # Determine which buttons can be used by agent
        game.add_available_button(Button.MOVE_RIGHT)
        game.add_available_button(Button.ATTACK)

        game.add_available_game_variable(GameVariable.AMMO2)       # Determine which game variables are in the state we get per timestep
        game.add_available_game_variable(GameVariable.POSITION_X)
        game.add_available_game_variable(GameVariable.POSITION_Y)

        game.set_episode_timeout(300)    # Episode timeout (should be 'max_episode_length' I think)
        game.set_episode_start_time(10)  # Episode start time (used to skip initial events like monsters spawning, weapon producing, etc)
        game.set_window_visible(False)   # Set visibility of game window
        game.set_sound_enabled(False)
        game.set_living_reward(-1)       # Set reward for each timestep
        game.set_mode(Mode.PLAYER)       # PLAYER mode lets agent perceive state and make actions (vs SPECTATOR mode)
        game.init()                      # Initialize game

        self.env = game

        # Return 3x3 identify matrix w/ True/False values in list form. I assume the repetition is an error...
        # This is DIFFERENT from "self.local_AC.actions" above and is used below in "r = self.env.make_action(self.actions[a]) / 100.0"
        # This must be how action choice is fed into Vizdoom engine
        #self.actions = self.actions = np.identity(a_size, dtype = bool).tolist()  # ORIGINAL CODE
        self.actions = np.identity(a_size, dtype = bool).tolist()  # MY CODE REMOVING REPETITION

    # Set up training process (trains network and updates weights). Called below in "work" function. "sess": Tensorflow session. "gamma": discount rate.
    # "rollout": list containing [initial state, action, reward, result state, done status, state value]
    # Not totally sure what the point of "bootstrap_value" is...I assume it's part of Generalized Advantage Estimation.
    # It's a value added to "rewards" and "values" and plays a part in computing advantages.
    # Below, it seems to be a sort of reward/value for the last state in a sequence of states.
    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]                 # Initial states
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        values = rollout[:, 5]                       # State values. "rollout[:, 4]" contains done statuses.

        # Use "rewards"/"values" to get advantage/discounted returns. Advantage function uses "Generalized Advantage Estimation".

        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])  # Adds value to "rewards". Not totally sure about "bootstrap_value", see blurb above

        # "discount": function above that computes total discounted reward for rest of episode from perspective of each state.
        # I think we remove last value to make dims work in "self.value_loss" above.
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]

        self.value_plus = np.asarray(values.tolist() + [bootstrap_value]) # Adds value to "values". Not totally sure about "bootstrap_value", see blurb above

        # Advantages can be defined as Q(s, a) - V(s), ie. diff btwn state/action value and state value.
        # Here, the 1st 2 terms approximate Q(s, a). They are state rewards + discounted state value of NEXT state.
        # Then subtract current state value. Dims work out since "value_plus" added "bootstrap_value" above.
        # See https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2 for more
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)  # Not totally sure about this, guess it's how Generalized Advantage Estimation works

        # Update global network using gradients from loss and generate network stats to periodically save
        feed_dict = {self.local_AC.target_v: discounted_rewards,      # Supply placeholders in "AC_Network()" class
                     self.local_AC.inputs: np.vstack(observations),
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages,
                     self.local_AC.state_in[0]: self.batch_rnn_state[0],   # supplied below
                     self.local_AC.state_in[1]: self.batch_rnn_state[1]}

        v_l, p_l, e_l, g_n, v_n, self.batch_rnn_state, _ = sess.run([self.local_AC.value_loss,      # "value_loss", etc are all quantities in "AC_Network" above
                                                                     self.local_AC.policy_loss,
                                                                     self.local_AC.entropy,         
                                                                     self.local_AC.grad_norms,      # Global norm of gradients
                                                                     self.local_AC.var_norms,       # Global norm of variables
                                                                     self.local_AC.state_out,
                                                                     self.local_AC.apply_grads],    # I assume this runs "apply_grads", not totally sure...
                                                                     feed_dict = feed_dict)

        return(v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n)   # Divide by "len(rollout)" to get sort of average value per state

    # Actually do training. "gamma": discount rate. "sess": Tensorflow session.
    # I think "max_episode_length" is the max timesteps in an episode.
    # I think this variable should be used in the "game.set_episode_timeout(300)" line above.
    # "coord" coordinates the multiple CPU threads being used ("tf.train.Coordinator()" below).
    # "saver": object to save/restore variables after training ("tf.train.Saver()" below).
    def work(self, max_episode_length, gamma, sess, coord, saver):

        episode_count = sess.run(self.global_episodes)  # These 2 things track different quantities
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():   # Sets default session and graph. Don't fully get this, but doesn't seem too important...

            # Not sure about details of this, but doesn't seem too important...
            while not coord.should_stop():       # "coord": coordinator for threads. "should_stop" checks if a stop was requested.

                sess.run(self.update_local_ops)  # Runs "update_target_graph" and sets worker network params to global network params

                episode_buffer = []     # Will be State, action, reward, new state, done status, state value
                episode_values = []
                episode_frames = []     # Holds episode game frames
                episode_reward = 0
                episode_step_count = 0
                d = False               # Episode done status
                self.env.new_episode()  # "self.env" = "game" above. Start new episode.

                # Get screen buffer (screenshot) pixels, part of the state. See http://vizdoom.cs.put.edu.pl/tutorial for more.
                s = self.env.get_state().screen_buffer   # Dims: 120 x 160
                episode_frames.append(s)
                s = process_frame(s)         # Function above that crops/resizes game frame. New dims: (7056, )

                # "rnn_state" is used as episode progresses. For training, I think you want a clean slate for 
                # RNN memory/output, that's why "train" function uses "batch_rnn_state". Not totally sure...
                rnn_state = self.local_AC.state_init  # "state_init" defined above as all 0s. "rnn_state" used as episode progresses
                self.batch_rnn_state = rnn_state  # "batch_rnn_state" used in "train" function

                while self.env.is_episode_finished() == False:
                    # Take action using probabilities from policy network output.
                    # Not updating weights here, so don't need to compute/apply gradients.
                    # "v": state value in 1 x 1 numpy array. "v[0, 0]" gets the value out.
                    a_dist, v, rnn_state = sess.run([self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                                                    feed_dict = {self.local_AC.inputs: [s],
                                                                 self.local_AC.state_in[0]: rnn_state[0],
                                                                 self.local_AC.state_in[1]: rnn_state[1]})

                    a = np.random.choice(a_dist[0], p = a_dist[0]) # Choose from actions in "a_dist" using "a_dist" as probabilities of choosing each action
                    a = np.argmax(a_dist == a)

                    # "self.actions" defined above. I don't know which entry of "self.actions" corresponds to which action.
                    # But don't think that matters, the network will learn the best action regardless.
                    # Not actually sure how to figure out what value of "self.actions" corresponds to which action...
                    r = self.env.make_action(self.actions[a]) / 100.0  # "make_action" takes action as input and returns reward
                    d = self.env.is_episode_finished()

                    if d == False:                                # Episode continues
                        s1 = self.env.get_state().screen_buffer   # Get new screen image pixels. "screen_buffer" is game screen.
                        episode_frames.append(s1)
                        s1 = process_frame(s1)
                    else:                                         # Episode ends
                        s1 = s

                    episode_buffer.append([s, a, r, s1, d, v[0, 0]]) # State, action, reward, new state, done status, state value
                    episode_values.append(v[0, 0])
                    episode_reward += r                     # Update quantities
                    s = s1
                    total_steps += 1
                    episode_step_count += 1

                    # If "episode_buffer" is a certain length and episode is not done or at max length, train and update global network.
                    # Not sure exactly why this condition is needed, or the value of "30" is used, or why last condition needs a "- 1"...
                    # I guess this mid-episode training helps somehow: perhaps "episode_buffer" getting too long is a concern, not sure...
                    # Maybe many small updates are preferred over fewer large updates...
                    if len(episode_buffer) == 30 and d != True and episode_step_count != max_episode_length - 1:

                        # Since we don't know the true final return (b/c it's mid-episode), "bootstrap" from current value estimation. [COMMENT FROM BLOG]
                        # "v1" used in "train" function below as bootstrap value. Not totally sure about bootstrap value, I wrote about it elsewhere...
                        v1 = sess.run(self.local_AC.value,                        # Get current state value
                                      feed_dict = {self.local_AC.inputs: [s],
                                                   self.local_AC.state_in[0]: rnn_state[0],
                                                   self.local_AC.state_in[1]: rnn_state[1]})[0, 0]    # "v1": state value in 1 x 1 numpy array

                        v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1)  # "v1" used as part of computing advantages
                        episode_buffer = []
                        sess.run(self.update_local_ops)

                    if d == True:   # If episode is over, break "while" statement
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))   # "episode_values" holds state values of episode states

                if len(episode_buffer) != 0:   # Update global network using episode buffer at end of episode
                    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0)

                # Periodically save gifs of episodes, model parameters, and summary stats
                if episode_count % 100 == 0 and episode_count != 0:
                    print(str(self.name) + ': ' + 'Episode Count: ' + str(episode_count))
                    if self.name == 'worker_0' and episode_count % 200 == 0:   # Periodically save episode GIF. I assume just for worker 0 to keep things manageable
                        time_per_step = 0.05  
                        images = np.array(episode_frames)
                        make_gif(images, './frames/image' + str(episode_count) + '.gif',  # Create and save GIF
                                 duration = len(images) * time_per_step, true_image = True, salience = False)

                    if episode_count % 500 == 0 and self.name == 'worker_0':  # Periodically save variables
                        saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                        print ("Saved Model: Epsiode " + str(episode_count))

                    mean_reward = np.mean(self.episode_rewards[-5:])     # Get recent rewards, episode lengths, average episode state values
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])  # "episode_mean_values" holds average of episode state values (1 value per episode)

                    summary = tf.Summary()    # Create summary to keep track of metrics
                    summary.value.add(tag = 'Performance/Reward', simple_value = float(mean_reward))
                    summary.value.add(tag = 'Performance/Length', simple_value = float(mean_length))
                    summary.value.add(tag = 'Performance/Value', simple_value = float(mean_value)) # average of recent episode state values
                    summary.value.add(tag = 'Losses/Value Loss', simple_value = float(v_l))
                    summary.value.add(tag = 'Losses/Policy Loss', simple_value = float(p_l))
                    summary.value.add(tag = 'Losses/Entropy', simple_value = float(e_l))
                    summary.value.add(tag = 'Losses/Grad Norm', simple_value = float(g_n))
                    summary.value.add(tag = 'Losses/Var Norm', simple_value = float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)
                    self.summary_writer.flush()      # Write event file to disk

                if self.name == 'worker_0':
                    sess.run(self.increment) # Defined above. Adds 1 to "global_episodes". Not sure why, see note above on "global_episodes"...
                episode_count += 1 # "episode_count" increases at end of every episode. "global_episodes" only increases for "worker_0".

##### TRAIN

max_episode_length = 300
gamma = 0.99                                                    # discount rate
s_size = 7056 # Screen pixels: observations are greyscale frames of 84 * 84 * 1
a_size = 3             # Action space size: agent can move Left, Right, or Fire
load_model = False                                  # Whether to load old model
model_path = './model'                            # Where to save model objects

tf.reset_default_graph()  # Not sure I get this 100%, but not too important...
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists('./frames'):  # Create directory to save episode playback gifs to
    os.makedirs('./frames')

with tf.device("/cpu:0"):  # Specify device to use for operations. Not sure I 100% get this, but doesn't seem too important.
    global_episodes = tf.Variable(0, dtype = tf.int32, name = 'global_episodes', trainable = False)
    trainer = tf.train.AdamOptimizer(learning_rate = 1e-4)

    # Generate global network. I assume global network doesn't need a trainer, because training is done by
    # workers and then global network is updated.
    master_network = AC_Network(s_size, a_size, 'global', None)

    #num_workers = multiprocessing.cpu_count() # Set workers to # of available CPU threads
    num_workers = 3
    workers = []

    for i in range(num_workers):                        # Create workers
        workers.append(Worker(DoomGame(), i, s_size, a_size, trainer, model_path, global_episodes)) # "DoomGame()" creates Doom game object

    saver = tf.train.Saver(max_to_keep = 5)      # Saves and restores variables

with tf.Session() as sess:
    coord = tf.train.Coordinator()  # A coordinator for threads, not totally sure why this is needed...

    if load_model == True:
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer()) # Initialize global variables

    # Asynchronous part. Start "work" process for each worker in a separate thread
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver) # Create lambda function w/ worker "work" function
        t = threading.Thread(target = (worker_work))          # Create a thread
        t.start()                                                # Start thread
        sleep(0.5)
        worker_threads.append(t)

    coord.join(worker_threads)  # Wait for threads to terminate. Not sure I totally get "coord" and threads, but doesn't seem too important...
