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
    xMin, xMax, yMin, yMax = 0,0,0,0

    indices_nonzero = edge_map.nonzero()
    # print(indices_nonzero)

    xMin = indices_nonzero[0].min()
    xMax = indices_nonzero[0].max()
    yMin = indices_nonzero[1].min()
    yMax = indices_nonzero[1].max()

    return xMin, xMax, yMin, yMax

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


##############################################################################
#                           Main script starts here                          #
##############################################################################

#filename = 'project_data/a/000224.png'
folder = 'project_data/a/'
newfolder = 'project_data/a_solution/'
edgefolder = 'project_data/a_edges/'
#filename = 'project_data/b/001319.png'
folder = 'project_data/b/'
newfolder = 'project_data/b_solution/'
edgefolder = 'project_data/b_edges/'

ransac_iterations = 100
ransac_threshold = 2
n_samples = 2

ratio = 0
i = -4

previous_m, previous_c = 0, 0

# Load the images and sort the paths to the sequenced names
filenames = imagesfilename_from_folder(folder)
filenames = np.sort(filenames)
#print(filenames)

# timestamp start
ts1 = time.time()
st = datetime.datetime.fromtimestamp(ts1).strftime('%Y-%m-%d %H:%M:%S')
print(st)

#for ind, filename in enumerate(filenames):
filename = filenames[0]
print(filename)
name, filetype = filename[:i], filename[i:]
newfilename = str(name)+"_solution"+str(filetype)
# print(""+str(name)+":::"+newfilename)

filepath = os.path.join(folder,filename)
newfilepath = os.path.join(newfolder, newfilename)
edgefilepath = os.path.join(edgefolder, filename)

image = plt.imread(filepath)
edges = edge_map(image)
#image = Image.open(filepath).convert('LA')



# detecing eyeCenter and size
xMin, xMax, yMin, yMax = min_max_eye(edges)
xEyeCenter, yEyeCenter = 348 , 191 # Center on first image of set A

yEyeCenter = (xMax+xMin)/2
xEyeCenter = (yMax+yMin)/2

if (xMax-xMin) > (yMax-yMin):
    xEyeSize = (xMax-xMin)/2
else:
    xEyeSize = (yMax-yMin)/2

#print("xmin:"+str(xMin)+"--"+str(xMax)+": ymin:"+str(yMin)+"--"+str(yMax))
# end of detecing eyeCenter and size

edgesAroundEye = edges[xMin:xMax,yMin:yMax]

f1 = plt.figure()
plt.imshow(edges)
#f1.savefig(edgefilepath, dpi=90, bbox_inches='tight')

plt.title('edge map')
f1.show()

plt.axis('off')
plt.show()


edge_pts = np.array(np.nonzero(edgesAroundEye), dtype=float).T
edge_pts_xy = edge_pts[:, ::-1]


# perform RANSAC iterations
for it in range(ransac_iterations):

    # this shows progress
    sys.stdout.write('\r')
    sys.stdout.write('iteration {}/{}'.format(it+1, ransac_iterations))
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

sys.stdout.write('\r')
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
print(st)

x = np.arange(image.shape[1])
y = model_m * x + model_c

print("m="+str(model_m)+"+c="+str(model_c)+"---Diff from previous m="+str(model_m-previous_m))
previous_m, previous_c = model_m, model_c

f2 = plt.figure()

#if m != 0 or c != 0:
#    plt.plot(x, y, 'r')

# plt.imshow(image)

xSolution, ySolution = 348 , 191 # 255, 140
h, w = 80, 80
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

print("xEyeCenter:"+str(xEyeCenter)+"-yEyeCenter-"+str(yEyeCenter)+": xEyeSize:"+str(xEyeSize))

ax1.add_patch(
    plt.Circle((xEyeCenter, yEyeCenter), radius= xEyeSize,
               edgecolor = 'red',
               facecolor="#00ccff",
               alpha=0.3
               )
)
plt.plot([xSolution], [ySolution], marker='o', markersize=5, color="red")

f2.show()
f2.savefig(newfilepath, dpi=90, bbox_inches='tight')

plt.close()

#plt.axis('off')
#plt.show()