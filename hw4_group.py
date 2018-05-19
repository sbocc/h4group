# import sys
import matplotlib
matplotlib.rcParams['image.cmap'] = 'gray'
matplotlib.rcParams['image.interpolation'] = 'nearest'
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
# from skimage import feature, color
from scipy.signal import convolve2d
import datetime
import time
from PIL import Image
import csv
# import cv2
from toolsHW2 import *
from toolsHW4 import *
from skimage.feature import match_template

##############################################################################
#                           Main script starts here                          #
##############################################################################

dataset = 0

if dataset == 1:
    #filename = 'project_data/a/000224.png'
    folder = 'project_data/a/'
    newfolder = 'project_data/a_solution/'
    edgefolder = 'project_data/a_edges/'
    xFirstSolution, yFirstSolution = 348, 191
    threshold = 0.4
else:
    #filename = 'project_data/b/001319.png'
    folder = 'project_data/b/'
    newfolder = 'project_data/b_solution/'
    edgefolder = 'project_data/b_edges/'
    xFirstSolution, yFirstSolution = 439, 272
    threshold = 0.25

xySolutions = []

ransac_iterations = 100
ransac_threshold = 2
n_samples = 2

ratio = 0
i = -4

previous_m, previous_c = 0, 0
xEyeCenter, yEyeCenter, xEyeSize = xFirstSolution,yFirstSolution,0
previous_xEyeCenter, previous_yEyeCenter, previous_xEyeSize = xEyeCenter, yEyeCenter, xEyeSize
first_xEyeSize = None

previous_texture_img = None
texture_patch = None
previous_texture_patch = None
previous_texture_patch_gen1 = None
previous_texture_patch_gen2 = None
first_texture_patch  = None

xSolution, ySolution = xFirstSolution , yFirstSolution
previous_xSolution, previous_ySolution = xFirstSolution, yFirstSolution
# Load the images and sort the paths to the sequenced names
filenames = imagesfilename_from_folder(folder)
filenames = np.sort(filenames)
#print(filenames)

# timestamp start
ts1 = time.time()
st = datetime.datetime.fromtimestamp(ts1).strftime('%Y-%m-%d %H:%M:%S')
print(st)

for index, filename in enumerate(filenames): # loop through all images in folder
#for index in range(0, 10): # loop only first 3
    filename = filenames[index]
    print(filename)
    name, filetype = filename[:i], filename[i:]
    newfilename = str(name)+"_solution"+str(filetype)
    # print(""+str(name)+":::"+newfilename)

    filepath = os.path.join(folder,filename)
    newfilepath = os.path.join(newfolder, newfilename)
    edgefilepath = os.path.join(edgefolder, filename)

    image = plt.imread(filepath)
    hist, bins = np.histogram(image.ravel(), 256, [0, 256], density=True)


    # ##################
    # Color or Grayscale
    # ##################
    dominant_channel = find_dominant_channel(image)[0] # set to -1 if you want color approach

    if dominant_channel > -1:
        image = image[:, :, dominant_channel]
    # end Color or Grayscale
    # ##################

    # ##################
    # Detect EyeCenter And Size
    # ##################
    previous_xEyeCenter, previous_yEyeCenter,previous_xEyeSize = xEyeCenter, yEyeCenter, xEyeSize
    # edge_pts, edge_pts_xy, \
    xEyeCenter, yEyeCenter, xEyeSize = detecingEyeCenterAndSize(image,threshold) # edges

    if first_xEyeSize is None:
        first_xEyeSize = xEyeSize

    # end Detect EyeCenter And Size
    # ##################

    # ##################
    # calculate RANSAC
    # ##################
    # model_m, model_c = performRANSAC(edge_pts, edge_pts_xy, ransac_iterations, ransac_threshold, n_samples, ratio)
    #
    # m = model_m
    # c = model_c
    # end calculate RANSAC
    # ##################

    sys.stdout.write('\r')
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    print(st)

    # ##################
    # show a line at the calculated RANSAC
    # ##################
    # x = np.arange(image.shape[1])
    # y = model_m * x + model_c
    # print("m="+str(model_m)+"+c="+str(model_c)+"---Diff from previous m="+str(model_m-previous_m))
    # previous_m, previous_c = model_m, model_c
    #
    # if m != 0 or c != 0:
    #     plt.plot(x, y, 'r')
    # end show a line at the calculated RANSAC
    # ##################

    f2 = plt.figure()

    # ##################
    # show the current image
    # ##################
    # plt.imshow(image)
    # end the current image
    # ##################

    square = int(first_xEyeSize / 2)
    h, w = square, square
    plus_divergent = 0
    gaussfilter = gauss2d(square, filter_size=square * 2)

    # ##################
    # pick a texture_img around the previous xySolution
    # ##################
    if dominant_channel > -1: # greyscale approach
        texture_img = image[ySolution + plus_divergent - h:ySolution + h + plus_divergent + 1,xSolution - w + plus_divergent:xSolution + w + plus_divergent + 1]
        #texture_rotateAndSum = rotateAndSumTexture(texture_img, dominant_channel)
        #textureCenter = texture_rotateAndSum[h][w]
        texture_filtered = texture_img * gaussfilter # gconv(texture_rotateAndSum, 1, square)
        #newShape = np.shape(texture_filtered)
        #entrypoint = int(square / 2)
        #texture_filtered = texture_filtered[entrypoint:entrypoint + 2 * h,entrypoint:entrypoint + 2 * w]
    else : # color approach
        texture_img = image[ySolution - h + plus_divergent:ySolution + h + plus_divergent,xSolution - w + plus_divergent:xSolution + w + plus_divergent, :]
        # texture_rotateAndSum = rotateAndSumTexture(texture_img, dominant_channel)
        # textureCenter = texture_rotateAndSum[h][w]
        # texture_filtered = texture_rotateAndSum
        texture_filtered = texture_img * gaussfilter
    # pick a texture_img around the previous xySolution
    ######

    #####
    # pick the matching texture_patch
    if previous_texture_img is None:
        previous_texture_img = texture_filtered.copy()
        entrypoint = int(square / 2)

    new_texture_patch = texture_filtered[entrypoint-5:entrypoint + h + 5,entrypoint - 5 :entrypoint + w + 5] * -1
    # end pick the maching texture_patch
    ######

    # ##################
    # create time_difference_texture_img as differenc from this to previous texture_img around the solution
    # ##################
    time_difference_texture_img = texture_filtered - previous_texture_img


    #time_difference_texture_img = gconv(time_difference_texture_img, 5, square)
    #newShape = np.shape(time_difference_texture_img)
    #entrypoint = int(square / 2)
    #time_difference_texture_img = time_difference_texture_img[entrypoint:entrypoint + 2 * h, entrypoint:entrypoint + 2 * w]
    # median = np.median(time_difference_texture_img)
    #time_difference_texture_img = rotateTexture180(time_difference_texture_img, dominant_channel)

    time_difference_texture_img = time_difference_texture_img * gaussfilter # multiply with a gausian around the center to lose uninteresting details at the border

    # end create time_difference_texture_img as differenc from this to previous texture_img around the solution
    # ##################

    # ##################
    # match the texture_patch to the selected time_difference_texture_img
    # ##################
    if index > 0:  # is not first image
        corr = match_template(time_difference_texture_img, texture_patch, pad_input=True)
        loc = tuple((np.where(corr == np.max(corr))[0][0], np.where(corr == np.max(corr))[1][0]))

        previous_xSolution, previous_ySolution = xSolution, ySolution
        xSolution, ySolution = xSolution - w + plus_divergent + loc[1], ySolution - h + plus_divergent + loc[0]

        ######
        # recalculate the sliding window variables
        texture_patch = (first_texture_patch + previous_texture_patch_gen1 + previous_texture_patch) / 3 # (new_texture_patch + previous_texture_patch) / 2
        previous_texture_patch_gen2 = previous_texture_patch_gen1
        previous_texture_patch_gen1 = previous_texture_patch
        previous_texture_patch = new_texture_patch
        # end recalculate the sliding window variables
        ######

    else: # is first image set sliding window the ground variables
        texture_patch = new_texture_patch
        first_texture_patch = new_texture_patch
        previous_texture_patch_gen2 = new_texture_patch
        previous_texture_patch_gen1 = new_texture_patch
        previous_texture_patch = new_texture_patch

    # end match the texture_patch to the selected time_difference_texture_img
    # ##################

    plt.imshow(image)
    plt.imshow(texture_patch)

    # ##################
    # paint a red square arount solution coordinate
    # ##################
    n_plots = 1
    size = np.shape(texture_patch)
    ax1 = plt.subplot(n_plots, n_plots, 1)
    ax1.add_patch(
        plt.Rectangle(
            (xSolution-size[0]/2, ySolution-size[1]/2),   # (x,y)
            size[0],          # width
            size[1],          # height
            edgecolor = 'red', # none
            facecolor="#00ffff",
            alpha=0.2
        )
    )
    # end paint a red square arount solution coordinate
    # ##################

    # ##################
    # paint a greed square arount solution coordinate
    # ##################
    n_plots = 1
    ax1 = plt.subplot(n_plots, n_plots, 1)
    ax1.add_patch(
        plt.Rectangle(
            (previous_xSolution - w + plus_divergent, previous_ySolution + plus_divergent - h),  # (x,y)
            w * 2,  # width
            h * 2,  # height
            edgecolor='green',  # none
            facecolor="#00ffff",
            alpha=0.2
        )
    )
    # end paint a greed square arount solution coordinate
    # ##################

    # ##################
    # paint a blue circle arount the area recognized as eye
    # ##################
    ax1.add_patch(
        plt.Circle((xEyeCenter, yEyeCenter), radius= xEyeSize,
                   edgecolor = 'red',
                   facecolor="#00ccff",
                   alpha=0.3
                   )
    )
    plt.plot([xSolution], [ySolution], marker='o', markersize=5, color="red")
    # end paint a blue circle arount the area recognized as eye
    # ##################

    # ##################
    # paint a blue circle arount the area previously recognized as eye
    # ##################
    ax1.add_patch(
        plt.Circle((previous_xEyeCenter, previous_yEyeCenter), radius=previous_xEyeSize,
                   edgecolor='red',
                   facecolor="#cccccc",
                   alpha=0.3
                   )
    )
    plt.plot([xSolution], [ySolution], marker='o', markersize=5, color="red")
    # end paint a blue circle arount the area previously recognized as eye
    # ##################


    # ##################
    # Store solution Coordinates and Save Image
    # ##################
    xySolutions.append([str(xSolution),str(ySolution)])

    f2.show()
    f2.savefig(newfilepath, dpi=90, bbox_inches='tight')

    plt.close()

    previous_texture_img = texture_img.copy()

b = open(newfolder+'solutions.csv', 'w')
a = csv.writer(b)
a.writerows(xySolutions)
b.close()

#plt.axis('off')
#plt.show()