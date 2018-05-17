####################################
#
# SIMPLEST APPROACH
#
# define a filter around the first point and use this to search on the following image
# assumption : from one image to the other, the shape, location and color of the tweezer does not change much

# loading modules
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
from skimage.feature import match_template

# loading images
# images a
# folder = "project_data/a"

# images b
folder = "project_data/b"

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = plt.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def imagespaths_from_folder(folder):
    imagesPaths = []
    for filename in os.listdir(folder):
        imagesPaths.append(os.path.join(folder,filename))
    return imagesPaths

def imagesfilename_from_folder(folder):
    filenames = []
    for filename in os.listdir(folder):
        if filename != ".DS_Store":
            filenames.append(filename)
    return filenames


# Load the images and sort the paths to the sequenced names
filenames = imagesfilename_from_folder(folder)
filenames = np.sort(filenames)
#print(filenames)
filename = filenames[0]
filepath = os.path.join(folder,filename)
filepath2 = os.path.join(folder, filenames[1])
img = plt.imread(filepath)
img2 = plt.imread(filepath2)
imgs = load_images_from_folder(folder)

# load coordinates of first point
# loc = (191, 348)  # images a
loc = (272, 439)  # images b


############################--------------------------------------------------------------------------------------------
# find patch around coordinate 1a in picture 1a
# duration : 3.5 min for 100 pictures

# timestamp start
ts1 = time.time()
st = datetime.datetime.fromtimestamp(ts1).strftime('%Y-%m-%d %H:%M:%S')
print(st)

# start with algo
img = plt.imread(os.path.join(folder, filenames[0]))
patch = img[(loc[0] - patch_half_size):(loc[0] + patch_half_size), (loc[1] - patch_half_size):(loc[1] + patch_half_size), :]
coordmax_list = list()
resimg_list = list()
patch_half_size = 50
for i in filenames[1:]:
    curr_img = plt.imread(os.path.join(folder, i))
    corr = match_template(curr_img, patch, pad_input=True)
    loc = tuple((np.where(corr == np.max(corr))[0][0], np.where(corr == np.max(corr))[1][0]))  #, np.where(corr == np.max(corr))[2][0]))
    coordmax_list.append((loc, i))
    plt.imshow(curr_img)
    plt.scatter(x=[loc[1]], y=[loc[0]], c='r', s=10)
    plt.savefig('b_simple_approach_filter_res/' + i)
    plt.clf()
    patch = curr_img[(loc[0] - patch_half_size):(loc[0] + patch_half_size), (loc[1] - patch_half_size):(loc[1] + patch_half_size), :]

# timestamp stop
sys.stdout.write('\r')
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
print(st)

# #######################
# # problem analysis:
# # at picture 42, the problem starts
# i = filenames[42]
# # to get coordinates from previous
# curr_img = plt.imread(os.path.join(folder, i))
# loc = coordmax_list[41][0]
# patch = curr_img[(loc[0] - patch_half_size):(loc[0] + patch_half_size), (loc[1] - patch_half_size):(loc[1] + patch_half_size), :]
# plt.imshow(patch)
#
#
# i = filenames[43]
# curr_img = plt.imread(os.path.join(folder, i))
# corr = match_template(curr_img, patch, pad_input=True)
# loc = tuple((np.where(corr == np.max(corr))[0][0],
#              np.where(corr == np.max(corr))[1][0]))  # , np.where(corr == np.max(corr))[2][0]))
# coordmax_list.append((loc, i))
# plt.imshow(curr_img)
# plt.scatter(x=[loc[1]], y=[loc[0]], c='r', s=10)
# plt.savefig('simple_approach_filter_res/' + i)
# plt.clf()
# patch = curr_img[(loc[0] - patch_half_size):(loc[0] + patch_half_size),
#         (loc[1] - patch_half_size):(loc[1] + patch_half_size), :]
#
# i = filenames[44]
# curr_img = plt.imread(os.path.join(folder, i))
# corr = match_template(curr_img, patch, pad_input=True)
# loc = tuple((np.where(corr == np.max(corr))[0][0],
#              np.where(corr == np.max(corr))[1][0]))  # , np.where(corr == np.max(corr))[2][0]))
# coordmax_list.append((loc, i))
# plt.imshow(curr_img)
# plt.scatter(x=[loc[1]], y=[loc[0]], c='r', s=10)
# plt.savefig('simple_approach_filter_res/' + i)
# plt.clf()
# patch = curr_img[(loc[0] - patch_half_size):(loc[0] + patch_half_size),
#         (loc[1] - patch_half_size):(loc[1] + patch_half_size), :]



############################--------------------------------------------------------------------------------------------
# improvement :
#
# sometimes one image is a bit crappy, therefore taking one after would help.
# or : take some candidate points.


############################--------------------------------------------------------------------------------------------
# improvement :
#
# sometimes one image is a bit crappy, therefore taking one after would help.
