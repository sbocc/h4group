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
dataset = 0  # 1 is a, 0 is b
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
# opatch = np.copy(patch)
cpatch = oimg[(loc[0] - patch_half_size):(loc[0] + patch_half_size), (loc[1] - patch_half_size):(loc[1] + patch_half_size)]
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
    col = 'r' # this match gives a red point in the resulting image
    if np.max(corr) < 0.8:
        print(i, "correlation lower than 0.8")
        # match with color patch from first image
        ccorr = match_template(curr_oimg, cpatch, pad_input=True)
        max_ccorr = np.max(ccorr)
        if max_ccorr > np.max(corr) :
            # if max of ccorr is higher than the one of corr, use ccorr to find location
            print("replace match with DoG with match with original color image")
            loc = tuple((np.where(ccorr == np.max(ccorr))[0][0], np.where(ccorr == np.max(ccorr))[1][0]))
            # make yellow dot in resulting image
            col = 'y'
        else:
            loc = tuple((np.where(corr == np.max(corr))[0][0], np.where(corr == np.max(corr))[1][0]))
    # ocorr = match_template(curr_img, opatch, pad_input=True)
    coordmax_list.append((loc, i, np.max(corr), patch, curr_img, curr_oimg))
    plt.imshow(curr_oimg)
    plt.scatter(x=[loc[1]], y=[loc[0]], c=col, s=10)
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





# # ERROR ANALYSIS -------------------------------------------------------------------------------------------------------
# # a
# # ------------------------------------------------------------------------------------------------------------------
# # if the correlation is below 0.8, use the patch from before image
# # coordmax_list[41] # what's 41 here is 42 in i
# patch = coordmax_list[44][3]
# # plt.imshow(cpatch)
# curr_img = coordmax_list[44][4]
# curr_oimg = coordmax_list[44][5]
# ocorr = coordmax_list[44][6]
# oloc = tuple((np.where(ocorr == np.max(ocorr))[0][0], np.where(ocorr == np.max(ocorr))[1][0]))
# corr = match_template(curr_img, patch, pad_input=True)
# ccorr = match_template(curr_oimg, cpatch, pad_input=True)
# np.max(ccorr)
# loc = tuple((np.where(corr == np.max(corr))[0][0], np.where(corr == np.max(corr))[1][0]))
# cloc = tuple((np.where(ccorr == np.max(ccorr))[0][0], np.where(ccorr == np.max(ccorr))[1][0]))
#
# plt.imshow(curr_oimg)
# plt.scatter(x=[loc[1]], y=[loc[0]], c='r', s=10)
# plt.scatter(x=[oloc[1]], y=[oloc[0]], c='g', s=10)
# plt.scatter(x=[cloc[1]], y=[cloc[0]], c='y', s=10)
#
#
#
# # idea : always check with original template. minimal match must be.
# # --> drawback: which treshold to use?
# # result : ?
