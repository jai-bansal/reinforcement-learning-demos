# FULL REVIEW DONE

import numpy as np
#import random
#import tensorflow as tf
#import matplotlib.pyplot as plt
#import scipy.misc
#import os
import csv
#import itertools
#import tensorflow.contrib.slim as slim

def processState(state1):                     # Function to reshape game frames
    return np.reshape(state1, [21168])

# Functions to update target network params with primary network params
# Below, variables are structured so the 1st half are from the primary network and 2nd half are from target network.
# "tau": rate to update target network toward primary network.
# Target network params slowly move towards primary network params.
def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars // 2]): # Loop through 1st half of variables

        # Update 2nd half of variables. Assign variable as a combo (based on "tau") of "var" and original variable value.
        # So what actually gets added to "op_holder"??
        op_holder.append(tfVars[idx + total_vars // 2].assign((var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))

    return(op_holder)

# I get the first 3 lines. The rest doesn't make sense to me...
def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)

    #total_vars = len(tf.trainable_variables())
    #a = tf.trainable_variables()[0].eval(session = sess)
    #b = tf.trainable_variables()[total_vars // 2].eval(session = sess)

    # ".all()" returns True when all elements are "True". If all numbers in "a" and "b" are valid,
    # you get "True" == "True", which is "True".
    # So this checks if running ".all()" on "a" and "b" returns the same results (not even necessarily "True").
    # Not sure what the point of this is...both "a.all()" and "b.all()" could return "False".
    # This does NOT appear to be checking that "a" and "b" are the same.
    #if a.all() == b.all():
    #    print("Target Set Success")
    #else:
    #    print("Target Set Failed")

# Function that allows GIFS of the training episode to be saved for use in Control Center.
# This function doesn't appear directly in "rl_6.py", it appears in the "saveToCenter" function below.
# It appears directly in "rl_8.py".

# "images": normal images. "fname": GIF file name. I think "duration" is the GIF length (in seconds).
# I think "true_image" refers to whether actual episode images are being used, vs...
# I think "salience" refers to whether salience "images" are being used.
# "salIMGS" are black/white images and end up being all black ("luminance").
# 1st, derivatives of advantage are taken per pixel and color channel.
# Then, the max derivative per pixel is taken (from the 3 color channel derivatives).
# This new thing is the "luminance" and is used as "salIMGS".
# It gives a visual idea of what pixels, if they were changed, would most change advantage.
def make_gif(images, fname, duration = 2, true_image = False, salience = False, salIMGS = None):
  import moviepy.editor as mpy  # video editing library

  # Function to make GIF frame at time "t". Pulls relevant image for time "t".
  # If creating true image [salience] GIF, returns that [a totally black] image.
  # I don't understand the mask computation or really how the salience images are constructed.
  # Just multiplying "imagesS" by 255 after scaling to 0/1 returns similar results to the original script.
  # That's how I would get the original results, no masking necessary.

  def make_frame(t):
    try:  # Pull relevant frame for GIF. Try a few numerical examples to understand this.
      x = images[int(len(images) / duration * t)]
    except:
      x = images[-1]                                          # Pull last frame

    if true_image:                            # If making actual episode GIF...
      return x.astype(np.uint8) # Return image. Not sure why data type is changed...
    else:                                           # If making salience GIF...
      # This turns image all black. Not sure why this exact form was used...
      # Maybe mask sits on top of this all black image, not totally sure...
      return ((x + 1) / 2 * 255).astype(np.uint8)

  # Function to make GIF mask at time "t". Only used for salience GIF.
  # Masking involves setting some pixel values in an image to zero, or some other "background" value.

  # I don't understand the mask computation or really how salience images are constructed.
  # Just multiplying "imagesS" by 255 after scaling to 0/1 returns similar results to the original script.
  # That's how I would get the original results, no masking necessary.
  def make_mask(t):
    try:  # Pull relevant salience frame for GIF. Try a few numerical examples to understand this.
      x = salIMGS[int(len(salIMGS) / duration * t)]
    except:
      x = salIMGS[-1]                                # Pull last salience frame
    return x

  # Create video clip w/ "make_frame" function and specified duration
  clip = mpy.VideoClip(make_frame, duration = duration)

  if salience == True: # If making salience GIF, generate frames using "make_mask" function

    # Create mask w/ "make_mask" function and specified duration
    mask = mpy.VideoClip(make_mask, ismask = True, duration = duration)

    # Command below fails, the salience GIFs are all black. Not really sure...
    # I don't understand the mask computation or really how salience images are constructed.
    # Just multiplying "imagesS" by 255 after scaling to 0/1 returns similar results to the original script.
    # That's how I would get the original results, no masking necessary.
    # mask = mpy.VideoClip(make_mask, ismask = False, duration = duration)

    #clipB = clip.set_mask(mask)  ## Should this be commented out?? Put mask on clip
    #clipB = clip.set_opacity(0)

    # Set opacity/transparency level of clip. I don't notice a major effect on salience GIFs when I comment it out.
    # Documentation: Returns a semi-transparent copy of the clip where the mask is multiplied by op (any float, normally between 0 and 1).
    #mask = mask.set_opacity(0.1)

    # Create GIF. "fname": filename. "fps": frames per second.
    # I would've thought we should write out "clipB", not the "mask"...not really sure...
    mask.write_gif(fname, fps = len(images) / duration, verbose = False)
    #clipB.write_gif(fname, fps = len(images) / duration,verbose=False)

  else:                        # Otherwise, generate frames using normal images
    # Create GIF. "fname": filename. "fps": frames per second.
    clip.write_gif(fname, fps = len(images) / duration, verbose = False)

# Record performance metrics and episode logs for Control Center.
# This function is called every "summaryLength" steps during training and testing in "rl_6.py".
# "i": episodes. "rList": list of total rewards per episode.
# "jList": list of steps per episode. "bufferArray": all experiences in episode.
# "summaryLength": number of episodes to periodically use for analysis.
# "h_size": size of final conv layer output. "sess": Tensorfow session.
# "mainQN": main Q-network. "time_per_step": length of each step in output GIFs.
def saveToCenter(i, rList, jList, bufferArray, summaryLength, h_size, sess, mainQN, time_per_step):
    with open('rl_6_model/log.csv', 'a') as myfile:
        state_display = (np.zeros([1, h_size]), np.zeros([1, h_size])) # initial state for computation below
        imagesS = []

        # "bufferArray[:, 0]" are original states in one episode. For each of these...
        for idx, z in enumerate(np.vstack(bufferArray[:, 0])):

            # Get salience and RNN state.
            # Salience: derivative of sum of "Advantage" wrt to each pixel in each image in "imageIn".
            # "salience" dims: (?, 84, 84, 3)
            # "salience" gives an idea of what pixels contribute the most to changing "Advantage".
            # A large [small] gradient for a pixel indicates that pixel changing would change "Advantage" a lot [little].
            # See "https://raghakot.github.io/keras-vis/visualizations/saliency/" for more info
            img, state_display = sess.run([mainQN.salience, mainQN.rnn_state],
                                          feed_dict = {mainQN.scalarInput: np.reshape(bufferArray[idx, 0], [1, 21168]) / 255.0,  # Scaled, reshape original states
                                                       mainQN.trainLength: 1,
                                                       mainQN.state_in: state_display,
                                                       mainQN.batch_size: 1})
            imagesS.append(img)

        # "imagesS" is a totally black image, with or without pixel scaling.
        # I don't understand the mask computation or really how the salience images are constructed.
        # Just multiplying "imagesS" by 255 after scaling to 0/1 returns similar results to the original script.
        # That's how I would get the original results, no masking necessary.
        imagesS = (imagesS - np.min(imagesS)) / (np.max(imagesS) - np.min(imagesS)) # Scale values
        imagesS = np.vstack(imagesS)
        imagesS = np.resize(imagesS, [len(imagesS), 84, 84, 3])

        # "salience": derivative per pixel per color channel. "imagesS": scaled saliences.
        # For each pixel, get max derivative among color channels.
        # "luminance" is a totally black image, with or without pixel scaling.
        # It gives a visual idea of what pixels, if they were changed, would most impact advantage.
        # I don't understand the mask computation or really how the salience images are constructed.
        # Just multiplying "imagesS" by 255 after scaling to 0/1 returns similar results to the original script.
        # That's how I would get the original results, no masking necessary.
        luminance = np.max(imagesS, 3)

        # "salience": derivative per pixel per color channel. "imagesS" was previously a bunch of scaled saliences.
        # For each pixel and color channel in "imagesS", get value of the max "salience" (derivative per pixel per color channel).
        # Pixels w/ low [high] gradients will be closer to value (0, 0, 0) [(1, 1, 1)] which is the color black [white].
        # Thse will be used to create salience GIF.
        # No effect when this is commented out.
        #imagesS = np.multiply(np.ones([len(imagesS), 84, 84, 3]),
        #                      np.reshape(luminance, [len(imagesS), 84, 84, 1]))

        # Create salience GIF. 1st argument is a black screen.
        # I don't understand the mask computation or really how the salience images are constructed.
        # Just multiplying "imagesS" by 255 after scaling to 0/1 returns similar results to the original script.
        # That's how I would get the original results, no masking necessary.
        make_gif(np.ones([len(imagesS), 84, 84, 3]),
                 './rl_6_model/frames/sal' + str(i) + '.gif',
                 duration = len(imagesS) * time_per_step,
                 true_image = False, salience = True, salIMGS = luminance)

        #images = zip(bufferArray[:, 0]) [ORIGINAL CODE]
        images = list(zip(np.array(bufferArray[:, 0]))) # MY CODE

        # Add resulting state of last action to "images".
        # Advantages are not computed for this frame, so it can't be included in the salience GIF.
        images.append(bufferArray[-1, 3])

        images = np.vstack(images)
        images = np.resize(images, [len(images), 84, 84, 3])

        # Create regular image GIF
        make_gif(images, './rl_6_model/frames/image' + str(i) + '.gif',
                 duration = len(images) * time_per_step, true_image = True,
                 salience = False)

        # return(images, imagesS, luminance) # MY CODE FOR UNDERSTANDING THIS FUNCTION

        # Write out results: episode, mean episode length of last "summaryLength" episodes,
        # mean episode reward of last "summaryLength" episodes, episode GIF path,
        # episode action tracker CSV path, episode salience GIF path.
        wr = csv.writer(myfile, quoting = csv.QUOTE_ALL)
        wr.writerow([i, np.mean(jList[-summaryLength:]), np.mean(rList[-summaryLength:]),
                     './rl_6_model/frames/image' + str(i) + '.gif',
                     './rl_6_model/frames/log' + str(i) + '.csv',
                     './rl_6_model/frames/sal' + str(i) + '.gif'])
        myfile.close()

    with open('./rl_6_model/frames/log' + str(i) + '.csv', 'w') as myfile: # Write out log file

        # At beginning of episode, there's no memory, so all zeros makes sense.
        state_train = (np.zeros([1, h_size]), np.zeros([1, h_size])) # initial state for computation below

        wr = csv.writer(myfile, quoting = csv.QUOTE_ALL)
        wr.writerow(["ACTION", "REWARD", "A0", "A1", 'A2', 'A3', 'V']) # Write header row

        # Get relative action advantages and state value for each state in episode
        a, v = sess.run([mainQN.Advantage, mainQN.Value],
                        feed_dict = {mainQN.scalarInput: np.vstack(bufferArray[:, 0]) / 255.0, # All scaled original states
                                     mainQN.trainLength: len(bufferArray),  # This is the number of states in the episode
                                     mainQN.state_in: state_train,
                                     mainQN.batch_size: 1})    # Only 1 episode

        # Why don't chosen actions always align with Q-values? Initially, all actions are random.
        # And even after, there's always a probability of random action.
        wr.writerows(zip(bufferArray[:, 1],                     # Write actions
                         bufferArray[:, 2],                     # Write rewards
                         a[:, 0], a[:, 1], a[:, 2], a[:, 3],    # Write relative action advantages
                         v[:, 0]))                              # Write state value
