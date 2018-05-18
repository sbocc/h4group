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
else:
    #filename = 'project_data/b/001319.png'
    folder = 'project_data/b/'
    newfolder = 'project_data/b_solution/'
    edgefolder = 'project_data/b_edges/'
    xFirstSolution, yFirstSolution = 439, 272

xySolutions = []

ransac_iterations = 100
ransac_threshold = 2
n_samples = 2

ratio = 0
i = -4

previous_m, previous_c = 0, 0

previous_texture_img = None
texture_patch = None
xSolution, ySolution = xFirstSolution , yFirstSolution

# Load the images and sort the paths to the sequenced names
filenames = imagesfilename_from_folder(folder)
filenames = np.sort(filenames)
#print(filenames)

# timestamp start
ts1 = time.time()
st = datetime.datetime.fromtimestamp(ts1).strftime('%Y-%m-%d %H:%M:%S')
print(st)

# for ind, filename in enumerate(filenames): # loop through all images in folder
for x in range(0, 10): # loop only first 3
    filename = filenames[x]
    #filename = filenames[0]
    print(filename)
    name, filetype = filename[:i], filename[i:]
    newfilename = str(name)+"_solution"+str(filetype)
    # print(""+str(name)+":::"+newfilename)

    filepath = os.path.join(folder,filename)
    newfilepath = os.path.join(newfolder, newfilename)
    edgefilepath = os.path.join(edgefolder, filename)

    image = plt.imread(filepath)

    dominant_channel = find_dominant_channel(image)[0]

    if dominant_channel > -1:
        image = image[:, :, dominant_channel]

    edges = edge_map(image)
    #image = Image.open(filepath).convert('LA') # greyscale convertion

    edge_pts, edge_pts_xy, xEyeCenter, yEyeCenter, xEyeSize = detecingEyeCenterAndSize(image, edges)

    # ##################
    # calculate RANSAC
    # ##################
    # model_m, model_c = performRANSAC(edge_pts, edge_pts_xy, ransac_iterations, ransac_threshold, n_samples, ratio)
    #
    # m = model_m
    # c = model_c
    # ##################
    # end
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
    # ##################
    # end ##############
    # ##################

    f2 = plt.figure()

    # ##################
    # show the current image
    # ##################
    # plt.imshow(image)
    # ##################
    # end ##############
    # ##################


    square = 50
    h, w = square, square
    plus_divergent = 0
    gaussfilter = gauss2d(square, filter_size=square * 2)

    if dominant_channel > -1:
        # texture_patch = image[ySolution + plus_divergent - int(h/2):ySolution + int(h/2) + plus_divergent + 1,xSolution - int(w/2) + plus_divergent:xSolution + int(w/2) + plus_divergent + 1]
        texture_img = image[ySolution + plus_divergent - h:ySolution + h + plus_divergent + 1,xSolution - w + plus_divergent:xSolution + w + plus_divergent + 1]
        #texture_rotateAndSum = rotateAndSumTexture(texture_img, dominant_channel)
        #textureCenter = texture_rotateAndSum[h][w]
        texture_filtered = texture_img * gaussfilter # gconv(texture_rotateAndSum, 1, square)
        #newShape = np.shape(texture_filtered)
        #entrypoint = int(square / 2)
        #texture_filtered = texture_filtered[entrypoint:entrypoint + 2 * h,entrypoint:entrypoint + 2 * w]

    else :
        texture_img = image[ySolution - h + plus_divergent:ySolution + h + plus_divergent,xSolution - w + plus_divergent:xSolution + w + plus_divergent, :]
        #texture_rotateAndSum = rotateAndSumTexture(texture_img, dominant_channel)
        #textureCenter = texture_rotateAndSum[h][w]
#        texture_filtered = texture_rotateAndSum
        texture_filtered = texture_img * gaussfilter

    if previous_texture_img is None:
        previous_texture_img = texture_filtered.copy()
        entrypoint = int(square / 2)
        texture_patch = texture_filtered[entrypoint:entrypoint + h,entrypoint:entrypoint + w] * -1

    time_difference_texture_img = texture_filtered - previous_texture_img
    #time_difference_texture_img = gconv(time_difference_texture_img, 5, square)
    #newShape = np.shape(time_difference_texture_img)
    #entrypoint = int(square / 2)
    #time_difference_texture_img = time_difference_texture_img[entrypoint:entrypoint + 2 * h, entrypoint:entrypoint + 2 * w]

    # median = np.median(time_difference_texture_img)
    #time_difference_texture_img = rotateTexture180(time_difference_texture_img, dominant_channel)
    time_difference_texture_img = time_difference_texture_img * gaussfilter
    # print(median)

    #if texture_patch is not None:
    if x > 0:  # ist not first image
        corr = match_template(time_difference_texture_img, texture_patch, pad_input=True)
        loc = tuple((np.where(corr == np.max(corr))[0][0], np.where(corr == np.max(corr))[1][0]))
        xSolution, ySolution = xSolution - w + plus_divergent + loc[1], ySolution - h + plus_divergent + loc[0]
        plt.scatter(x=[loc[1]], y=[loc[0]], c='g', s=10)


    loc = tuple((ySolution,xSolution))
    plt.scatter(x=[loc[1]], y=[loc[0]], c='g', s=10)

    # time_difference_texture_img[time_difference_texture_img == nan] = 0
    # print("textureCenter x:" + str(h) + "-y-" + str(w) + ": textureCenter:" + str(textureCenter)+ ": textureCenter shape:" + str(np.shape(texture_filtered)))

    # print( "textureCenter:" + str(textureCenter))
    # textureCenter = texture_filtered[h][w]
    #texture_filtered[texture_filtered < textureCenter - 0.1] = 0
    #texture_filtered[texture_filtered > textureCenter + 0.1] = 0
    #

    # print("texture_filtered:" + str(texture_filtered))
    # texture_filtered = gconv(texture_filtered, 5 , square)

    # texture_filtered = texture_filtered.nonzero()

    #texture_edges = edge_map(texture_img)
    #texture_edges = edge_map(texture_rotateAndSum)
    #texture_edges_trans = np.transpose(texture_edges)
    #img_filtered = convolve2d(image, np.transpose(image))

    # print("xSolution:" + str(xSolution - w) + "-ySolution-" + str(xSolution + w) + ": xEyeSize:" + str(np.shape(texture_img)))

    plt.imshow(image)
    #plt.scatter(x=[h], y=[w], c='r', s=10)

    #plt.show()

    # ##################
    # paint a red square arount solution coordinate
    # ##################
    n_plots = 1
    ax1 = plt.subplot(n_plots, n_plots, 1)
    ax1.add_patch(
        plt.Rectangle(
            (xSolution-w/2, ySolution-h/2),   # (x,y)
            w,          # width
            h,          # height
            edgecolor = 'red', # none
            facecolor="#00ffff",
            alpha=0.2
        )
    )
    # ##################
    # end ##############
    # ##################

    # ##################
    # paint a blue circle arount the area recognized as eye
    # ##################
    # print("xEyeCenter:"+str(xEyeCenter)+"-yEyeCenter-"+str(yEyeCenter)+": xEyeSize:"+str(xEyeSize))
    # ax1.add_patch(
    #     plt.Circle((xEyeCenter, yEyeCenter), radius= xEyeSize,
    #                edgecolor = 'red',
    #                facecolor="#00ccff",
    #                alpha=0.3
    #                )
    # )
    # plt.plot([xSolution], [ySolution], marker='o', markersize=5, color="red")
    # ##################
    # end ##############
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