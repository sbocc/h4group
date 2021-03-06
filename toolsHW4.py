""" 
Contains functions that are mainly used in all exercises of HW2
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
import time
from scipy.misc import imresize
from scipy.signal import convolve2d
import scipy.signal as signal
from itertools import chain
from skimage import feature, color
import os
import sys
from skimage.feature import match_template

################
# Load images from Folder
################

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

def min_max_eye(edge_map):
    indices_nonzero = edge_map.nonzero()

    xMin = indices_nonzero[0].min()
    xMax = indices_nonzero[0].max()
    yMin = indices_nonzero[1].min()
    yMax = indices_nonzero[1].max()

    return xMin, xMax, yMin, yMax

#####################################
# Find dominant color channel
#####################################
def find_dominant_channel(img):
    # find out which color channel is dominant
    # check for each channel if its mean is higher than the mean before
    om = 0
    for i in range(np.shape(img)[2]):
        nm = np.mean(img[:, :, i])
        if nm > om:
            om = nm
    return((i, om))


##############################################################################
#                        Functions to complete                               #
##############################################################################

################
# EXERCISE 1.1 #
################


def edge_map(img):
    # Returns the edge map of a given image.
    #
    # Inputs:
    #   img: image of shape (n, m, 3) or (n, m)
    #
    # Outputs:
    #   edges: the edge map of image

    #
    # REPLACE THE FOLLOWING WITH YOUR CODE
    #
    #edges = np.zeros(img.shape[0:2])
    #edges[np.random.randint(image.shape[0], size=100), np.random.randint(image.shape[1], size=100)] = 1

    #sobel_filter = np.asarray([1, 0, -1, 2, 0, -2, 1, 0, -1]).reshape((1, 9))
    #prewitt_filter = np.asarray([1, 0, -1, 1, 0, -1, 1, 0, -1]).reshape((1, 9))
    #roberts_cross_filter = np.asarray([-1, 0, 0, 1]).reshape((1, 4))
    #laplacian_filter = np.asarray([0, -1, 0, -1, 4, -1, 0, -1, 0]).reshape((1, 9))
    #edges = convolve2d(img, laplacian_filter, mode='full', boundary='fill', fillvalue=0)

    grayimg = color.rgb2gray(img)
    edges = feature.canny(grayimg,sigma=1.5, low_threshold=None, high_threshold=None, mask=None, use_quantiles=False)

    return edges


################
# EXERCISE 1.2 #
################


def fit_line(points):
    # Fits a line y=m*x+c through two given points (x0,y0) and
    # (x1,y1). Returns the slope m and the y-intersect c of the line.
    #
    # Inputs:
    #   points: list with two 2D-points [[x0,y0], [x1,y1]]
    #           where x0,y0,x0,y1 are integers
    #
    # Outputs:
    #   m: the slope of the fitted line, integer
    #   c: the y-intersect of the fitted line, integers
    #
    # WARNING: vertical and horizontal lines should be treated differently
    #          here add some noise to avoid division by zero.
    #          You could use for example sys.float_info.epsilon

    #
    # REPLACE THE FOLLOWING WITH YOUR CODE
    #

    if (len(points) < 2):
        m = 0
        c = 0

        return m, c


    x0 = points[0][0]
    y0 = points[0][1]
    x1 = points[1][0]
    y1 = points[1][1]
    #print('x0:'+str(x0)+' y0:'+str(y0))
    #print('x1:'+str(x1)+' y1:'+str(y1))

    # caluculate y = m * x + c
    if (x1 - x0) == 0 :
        m = 0
        c = 0
        return m, c

    # y1 = m * x1 + c

    m = (y1 - y0) / (x1 - x0)
    c = y1 - m * x1

    # print('m:'+str(m)+' c:'+str(c))

    return m, c


################
# EXERCISE 1.3 #
################

def quadrat(x):
    return x ** 2

def point_to_line_dist(m, c, x0, y0):
    # Returns the minimal distance between a given
    #  point (x0,y0)and a line y=m*x+c.
    #
    # Inputs:
    #   x0, y0: the coordinates of the points
    #   m, c: slope and intersect of the line
    #
    # Outputs:
    #   dist: the minimal distance between the point and the line.

    #
    # REPLACE THE FOLLOWING WITH YOUR CODE
    #
    dist = 0

    if m * x0 + c == y0 : # Point is already on the line
        return dist

    # d(x,y) = sqr( (x-x0)^2 + (y-y0)^2 )
    # d'(x) = (2*(x-x0) + 2*(f(x)-y0) * f'(x)
    # f(x) = m * x + c -> f'(x) = m
    # 0 = 2*(x - x0) + 2 * ( m * x + c - y0) * m
    # 0 = 2x - 2* x0 + 2 * m^2 * x + 2 * c *m  - 2 * y0 * m
    # 2 * y0 * m - 2 * c *m + 2* x0 = 2x + 2 * m^2 * x = 2x (1 + m^2)

    x_onLine = (y0 * m - c * m + x0) / (1 + m * m)
    y_onLine = m * x_onLine + c
    #print('x_onLine:' + str(x_onLine) + ' y_onLine:' + str(y_onLine))

    dist = np.sqrt(quadrat(x0 - x_onLine) + quadrat(y0 - y_onLine))
    return dist


def detecingEyeCenterAndSize(image, threshold = 0.4): # edges,
    # detecing eyeCenter and size
    calcimage = image.copy()
    calcimage[calcimage < threshold] = 0

    xMin, xMax, yMin, yMax = min_max_eye(calcimage)
    xEyeCenter, yEyeCenter = None, None

    yEyeCenter = int((xMax + xMin) / 2)
    xEyeCenter = int((yMax + yMin) / 2)

    if (xMax - xMin) > (yMax - yMin):
        xEyeSize = int((xMax - xMin) / 2)
    else:
        xEyeSize = int((yMax - yMin) / 2)

    #edgesAroundEye = edges[xMin:xMax, yMin:yMax]

    #edge_pts = np.array(np.nonzero(edgesAroundEye), dtype=float).T
    # edge_pts_xy = edge_pts[:, ::-1]
    # edge_pts, edge_pts_xy,
    return  xEyeCenter, yEyeCenter, xEyeSize


def performRANSAC(edge_pts, edge_pts_xy, ransac_iterations, ransac_threshold, n_samples, ratio):
    # perform RANSAC iterations
    model_m = 0
    model_c = 0
    # global model_m, model_c
    for it in range(ransac_iterations):

        # this shows progress
        sys.stdout.write('\r')
        sys.stdout.write('iteration {}/{}'.format(it + 1, ransac_iterations))
        sys.stdout.flush()

        all_indices = np.arange(edge_pts.shape[0])
        np.random.shuffle(all_indices)

        indices_1 = all_indices[:n_samples]
        indices_2 = all_indices[n_samples:]

        maybe_points = edge_pts_xy[indices_1, :]
        test_points = edge_pts_xy[indices_2, :]

        # find a line model for these points
        m, c = fit_line(maybe_points)

        x_list = []
        y_list = []
        num = 0

        # find distance to the model for all testing points
        for ind in range(test_points.shape[0]):

            x0 = test_points[ind, 0]
            y0 = test_points[ind, 1]

            # distance from point to the model
            dist = point_to_line_dist(m, c, x0, y0)

            # check whether it's an inlier or not
            if dist < ransac_threshold:
                num += 1

        # in case a new model is better - cache it
        if num / float(n_samples) > ratio:
            ratio = num / float(n_samples)
            model_m = m
            model_c = c

    return model_m, model_c

################
# EXERCISE 4.1 #
################

def mirrorTextureAt45Degree(texture, dominant_channel = -1):
    if dominant_channel > -1:
        tex_rows, tex_cols = np.shape(texture)
    else:
        tex_rows, tex_cols, tex_bands = np.shape(texture)

    assert tex_rows > 0 and tex_cols > 0 and tex_rows == tex_cols, "texture is not a square"

    rotatedTexture = texture.copy() # np.zeros((tex_rows, tex_cols))

    # print("ssd:"+str(ssd))
    # for ind, value in np.ndenumerate(rotatedTexture):
    #
    # ADD YOUR CODE HERE
    #
    for i in range(tex_rows):
        for j in range(tex_cols):
            rotatedTexture[i, j] = rotatedTexture[i, j] + texture[j,i]

    return rotatedTexture

def rotateTexture90(texture, dominant_channel = -1):
    if dominant_channel > -1:
        tex_rows, tex_cols = np.shape(texture)
    else:
        tex_rows, tex_cols, tex_bands = np.shape(texture)

    assert tex_rows > 0 and tex_cols > 0 and tex_rows == tex_cols, "texture is not a square"

    rotatedTexture = texture.copy() # np.zeros((tex_rows, tex_cols))

    # print("ssd:"+str(ssd))
    # for ind, value in np.ndenumerate(rotatedTexture):
    #
    # ADD YOUR CODE HERE
    #
    for i in range(tex_rows):
        for j in range(tex_cols):
            rotatedTexture[i, j] = texture[j, tex_cols - i - 1]

    return rotatedTexture

def rotateTexture180(texture, dominant_channel = -1):
    if dominant_channel > -1 :
        tex_rows, tex_cols = np.shape(texture)
    else:
        tex_rows, tex_cols, tex_bands = np.shape(texture)

    assert tex_rows > 0 and tex_cols > 0 and tex_rows == tex_cols, "texture is not a square"

    rotatedTexture = texture.copy() # np.zeros((tex_rows, tex_cols))

    # print("ssd:"+str(ssd))
    # for ind, value in np.ndenumerate(rotatedTexture):
    #
    # ADD YOUR CODE HERE
    #
    for i in range(tex_rows):
        for j in range(tex_cols):
            # rotatedTexture[i, j] = texture[tex_rows - i - 1, tex_cols - j - 1]
            rotatedTexture[i, j] = texture[i, j] + texture[tex_rows - i - 1, tex_cols - j - 1]

    return rotatedTexture

def rotateAndSumTexture(texture, dominant_channel = -1):

    if dominant_channel > -1 :
        tex_rows, tex_cols = np.shape(texture)
    else:
        tex_rows, tex_cols, tex_bands = np.shape(texture)

    assert tex_rows > 0 and tex_cols > 0 and tex_rows == tex_cols, "texture is not a square"

    rotatedTexture = texture.copy()  # np.zeros((tex_rows, tex_cols))

    # print("ssd:"+str(ssd))
    # for ind, value in np.ndenumerate(rotatedTexture):
    #
    # ADD YOUR CODE HERE
    #
    for i in range(tex_rows):
        for j in range(tex_cols):
            # rotation und aufsummieren gegen Uhrzeiger (nach links)
            # rotatedTexture[i, j] = (texture[i, j] + texture[j, tex_cols - i - 1]) / 2
            # rotation und aufsummieren im Uhrzeiger (nach recht)
             rotatedTexture[i, j] = (texture[i, j] + texture[tex_rows - j - 1, i]) / 2
            # rotation und aufsummieren (nach links und recht)
            # rotatedTexture[i, j] = (texture[i, j] + texture[tex_rows - j - 1, i] + texture[j, tex_cols - i - 1]) / 3
            # rotation und aufsummieren (nach links, recht und 180)
            #rotatedTexture[i, j] = (texture[i, j] * texture[i, j] + texture[tex_rows - j - 1, i] * texture[tex_rows - j - 1, i] + texture[j, tex_cols - i - 1] * texture[j, tex_cols - i - 1] + texture[tex_rows - i - 1, tex_cols - j - 1] * texture[tex_rows - i - 1, tex_cols - j - 1]) / 4
            # rotation und aufsummieren (nur 180)
            #rotatedTexture[i, j] = (texture[i, j] * texture[i, j] + texture[tex_rows - i - 1, tex_cols - j - 1] * texture[tex_rows - i - 1, tex_cols - j - 1]) / 2

    return rotatedTexture

def match_texture_patch(new_texture_patch,previous_texture_patch):
    newShape = np.shape(new_texture_patch)
    previousShape = np.shape(previous_texture_patch)
    shapeCenterX = int(previousShape[0] / 2)
    shapeCenterY = int(previousShape[1] / 2)

    corr = match_template(previous_texture_patch, new_texture_patch, pad_input=True)
    loc = tuple((np.where(corr == np.max(corr))[0][0], np.where(corr == np.max(corr))[1][0]))

    # print("newShape:"+str(newShape)+" previousShape:"+str(previousShape)+" loc:"+str(loc))
    # print("newlocation is x:"+str(loc[1]-shapeCenterX)+" y:"+str(loc[0]-shapeCenterY))

    return tuple((loc[0] - shapeCenterY, loc[1] - shapeCenterX))

def calculate_RANSAC_onEye(eyeImg):
    edges = feature.canny(eyeImg, sigma=1.5, low_threshold=None, high_threshold=None, mask=None, use_quantiles=False)
    # edges = edge_map(eyeImg)

    plt.subplot(1,1,1)
    plt.imshow(edges)
    plt.show()

    #f1 = plt.figure()
    #plt.imshow(edges)
    #f1.show()

    edge_pts = np.array(np.nonzero(edges), dtype=float).T
    edge_pts_xy = edge_pts[:, ::-1]

    ransac_iterations = 50
    ransac_threshold = 2
    n_samples = 2

    ratio = 0

    # ##################
    # calculate RANSAC
    # ##################
    model_m, model_c = performRANSAC(edge_pts, edge_pts_xy, ransac_iterations, ransac_threshold, n_samples, ratio)
    m = model_m
    c = model_c
    # end calculate RANSAC
    # ##################

    x = np.arange(eyeImg.shape[1])
    y = model_m * x + model_c

    f2 = plt.figure()

    if m != 0 or c != 0:
        plt.plot(x, y, 'r')
    plt.imshow(eyeImg)

    f2.show()

    plt.show()

    return eyeImg