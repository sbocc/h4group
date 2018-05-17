############################
# 1 - loading data
# 2 - idea 1
#

import sys
import matplotlib
matplotlib.rcParams['image.cmap'] = 'gray'
matplotlib.rcParams['image.interpolation'] = 'nearest'
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from skimage import feature, color
from scipy.signal import convolve2d
import datetime
import time
import os
from PIL import Image
import csv
from toolsHW2 import *
from toolsHW4 import *

dataset = 1

if dataset == 0:
    folder = 'project_data/a/'
else:
    folder = 'project_data/b/'


imgs = load_images_from_folder(folder)
#plt.imshow(imgs[2])
test = np.copy(imgs[1])
tesw = np.copy(imgs[8][:,:,1])

# run code from homework 2!

# - 2 - ##################
# Idea 1
#############
# idea :  don't detect the tool itself, but the deviation from a clear round shape.
# approach : ?

# ------------------ testing around
# where are high differences in illumination? Find the border pixels.
# make image lighter, enhance contrast


np.min(imgs[1])
np.max(imgs[1])
np.mean(imgs[1])
np.median(imgs[1])

test[test > np.mean(imgs[1])] = 1
plt.imshow(test[:,:,1])
plt.imshow(tesw)

plt.hist(tesw.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
tesw[tesw < 0.3] = 1
tesw = tesw + 0.2
tesw[tesw > 1] = 1
np.max(tesw)

# - 3 - ##################
# Idea 2
#############
# APPLY DIFFERENCE OF GAUSSIAN
DoG_of_img = DoG(tesw, 10, 25, 5)
plt.imshow(DoG_of_img)

# - 4 - ##################
# Idea 3
#############
# subtract a smoothed image to get the shape much better
img_filtered_1 = gconv(tesw, 70, 41)
np.shape(img_filtered_1)
tesw0 = np.pad(tesw, 20, mode="constant")
np.shape(tesw0)
plt.imshow(img_filtered_1)
plt.imshow(tesw0)
gagi = tesw0 - img_filtered_1
plt.imshow(gagi)

plt.show()

# next thing to do : run a detector of a "stab" here! a filter.
# maybe detection on this works best.

