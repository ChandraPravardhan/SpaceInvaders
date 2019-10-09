import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
import retro                 # Retro Environment


from skimage import transform          # Help us to preprocess the frames
from skimage.color import rgb2gray     # Help us to gray our frames

import matplotlib.pyplot as plt        # Display graphs

from collections import deque          # Ordered collection with ends

import random

import warnings                        # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')
                                       # Create our environment
env = retro.make(game='SpaceInvaders-Atari2600')

print("The size of our frame is: ", env.observation_space)
print("The action size is : ", env.action_space.n)

# Here we create an hot encoded version of our actions
# possible_actions = [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0]...]
possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())
