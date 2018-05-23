####################################
#
# DIFFERENCE OF GAUSSIAN APPROACH
#
# define a filter around the first point and use this to search on the following image
# assumption : from one image to the other, the shape, location and color of the tweezer does not change much
# SIMPLIFICATION   : only focus on the dominant color channel to be faster.
# IDEA : instead of only taking the dominant color channel, we substract the dominant color channel from a smoothed image
#    to get a more detailed picture for searching.
# time : 2min 45 for a; 2min 52 for b

# ----------------------------------------------------------------------------------------------------------------------
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
from toolsHW4 import *

# define variables for filtering
sigma = 70
filter_size = 41 # must be an uneven number!
filter_half_size = int((filter_size - 1) / 2)

# loading images
dataset = 1  # 1 is a, 0 is b
if dataset == 1:
    folder = 'project_data/a/'
    xFirstSolution, yFirstSolution = 348, 191
    newfolder = 'project_data/a_DoG_filter_res/'
    patch_half_size = 50
    name = 'a'
else:
    folder = 'project_data/b/'
    xFirstSolution, yFirstSolution = 439, 272
    newfolder = 'project_data/b_DoG_filter_res/'
    patch_half_size = 20
    name = 'b'

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
loc = (yFirstSolution, xFirstSolution)

# ---------------------------------------------------------------------------------------------------------------------

# oimg is the original image, img will be the difference of gaussian to work with.
# the difference of gaussian image is (filter_size -1 ) / 2 pixels longer/higer.
oimg = plt.imread(os.path.join(folder, filenames[0]))
img = oimg[:, :, find_dominant_channel(oimg)[0]]
img_1 = gconv(img, sigma, filter_size)
img_1 = img_1[filter_half_size:-filter_half_size, filter_half_size:-filter_half_size]
img = img - img_1
# take out the padding
patch = img[(loc[0] - patch_half_size):(loc[0] + patch_half_size), (loc[1] - patch_half_size):(loc[1] + patch_half_size)]

# timestamp start
ts1 = time.time()
st = datetime.datetime.fromtimestamp(ts1).strftime('%Y-%m-%d %H:%M:%S')
print(st)

# list of coordinates with maximum match
coordmax_list = list()
# list of resulting images
resimg_list = list()
for i in filenames[1:]:
    curr_oimg = plt.imread(os.path.join(folder, i))
    curr_img = curr_oimg[:, :, find_dominant_channel(curr_oimg)[0]]
    curr_img_1 = gconv(curr_img, sigma, filter_size)
    curr_img_1 = curr_img_1[filter_half_size:-filter_half_size, filter_half_size:-filter_half_size]
    curr_img = curr_img - curr_img_1
    corr = match_template(curr_img, patch, pad_input=True)
    loc = tuple((np.where(corr == np.max(corr))[0][0], np.where(corr == np.max(corr))[1][0]))
    coordmax_list.append((loc, i, np.max(corr), patch, curr_img, curr_oimg))
    plt.imshow(curr_oimg)
    plt.scatter(x=[loc[1]], y=[loc[0]], c='r', s=10)
    plt.savefig(newfolder + i)

    plt.clf()
    patch = curr_img[(loc[0] - patch_half_size):(loc[0] + patch_half_size), (loc[1] - patch_half_size):(loc[1] + patch_half_size)]

# timestamp stop
sys.stdout.write('\r')
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
print(st)

b = open(newfolder+'solutions.csv', 'w')
a = csv.writer(b)
a.writerows(coordmax_list)
b.close()

# ERROR ANALYSIS
# a
# from picture 000267.png, there is an error:
i = filenames[44]
sorted = np.sort(corr, axis = None)
np.shape(sorted)
sorted[307180:307200]
# the maximum corr value of the problem image is too low.
plt.imshow(patch)
plt.imshow(curr_img)

# test if the color template would fit better
curr_oimg # is the image to search the template
loc = (192, 359) # location of the previous

patcho = curr_oimg[(loc[0] - patch_half_size):(loc[0] + patch_half_size), (loc[1] - patch_half_size):(loc[1] + patch_half_size), :]# define the color image
test1 = match_template(curr_oimg, patcho)
np.max(test1)
loc1 = tuple((np.where(test1 == np.max(test1))[0][0], np.where(test1 == np.max(test1))[1][0]))
plt.imshow(patch66)
plt.scatter(x=[loc1[1]], y=[loc1[0]], c='r', s=10)

# take patch from 266 and fit to 268
patch66 = np.copy(patch)
loc
np.max(corr) # 0.77

# if the correlation is below 0.8, don't search on the given image and go to next
i = filenames[42]
# run stuff
np.max(corr)
i = filenames[44]
# now comes the bad image
if np.max(corr) < 0.8:
    # keep the patch for the next image
    patcho = np.copy(patch)
    # later : find whatever can be found in this image (even if it's wrong)
    # go to next image (without having saved a new patch
i = filenames[45]
plt.imshow(patch)
i = filenames[46]

# if the correlation is below 0.8, use the patch from before image
coordmax_list[41] # what's 41 here is 42 in i
patch = coordmax_list[42][3]
curr_img = coordmax_list[43][4]
corr = match_template(curr_img, patch, pad_input=True)
np.max(corr)
loc = tuple((np.where(corr == np.max(corr))[0][0], np.where(corr == np.max(corr))[1][0]))

plt.imshow(curr_oimg)
plt.scatter(x=[loc[1]], y=[loc[0]], c='r', s=10)


test = list()
for i in range(0,len(coordmax_list)):
    test.append(coordmax_list[i][2])

plt.hist(test)
type(test)
test = np.array(test)
np.where(test < 0.8)

coordmax_list[42][]

# entweder ist patch oder curr_img scheisse.
# welches?
# coordmax_list[41] # what's 41 here is 42 in i
patch = coordmax_list[42][3]
curr_img = coordmax_list[44][4]
corr = match_template(curr_img, patch, pad_input=True)
np.max(corr)
loc = tuple((np.where(corr == np.max(corr))[0][0], np.where(corr == np.max(corr))[1][0]))

plt.imshow(curr_img)
plt.scatter(x=[loc[1]], y=[loc[0]], c='r', s=10)

testo = np.mean([coordmax_list[42][3], coordmax_list[41][3]], axis=(0,1))
testom = (testo - np.min(testo)) / (np.max(testo) - np.min(testo))
plt.imshow(testom)

plt.imshow(patch)
plt.imshow(curr_img)

