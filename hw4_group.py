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

##############################################################################
#                           Main script starts here                          #
##############################################################################

dataset = 0

if dataset == 0:
    #filename = 'project_data/a/000224.png'
    folder = 'project_data/a/'
    newfolder = 'project_data/a_solution/'
    edgefolder = 'project_data/a_edges/'
else:
    #filename = 'project_data/b/001319.png'
    folder = 'project_data/b/'
    newfolder = 'project_data/b_solution/'
    edgefolder = 'project_data/b_edges/'

xySolutions = []

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

# for ind, filename in enumerate(filenames): # loop through all images in folder
for x in range(0, 3): # loop only first 3
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
    edges = edge_map(image)
    #image = Image.open(filepath).convert('LA') # greyscale convertion

    edge_pts, edge_pts_xy, xEyeCenter, yEyeCenter, xEyeSize = detecingEyeCenterAndSize(image, edges)

    model_m, model_c = performRANSAC(edge_pts, edge_pts_xy, ransac_iterations, ransac_threshold, n_samples, ratio)

    m = model_m
    c = model_c

    sys.stdout.write('\r')
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    print(st)

    x = np.arange(image.shape[1])
    y = model_m * x + model_c

    print("m="+str(model_m)+"+c="+str(model_c)+"---Diff from previous m="+str(model_m-previous_m))
    previous_m, previous_c = model_m, model_c

    f2 = plt.figure()

    if m != 0 or c != 0:
        plt.plot(x, y, 'r')

    plt.imshow(image)

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

    xySolutions.append([str(xSolution),str(ySolution)])

    f2.show()
    f2.savefig(newfilepath, dpi=90, bbox_inches='tight')

    plt.close()

b = open(newfolder+'solutions.csv', 'w')
a = csv.writer(b)
a.writerows(xySolutions)
b.close()

#plt.axis('off')
#plt.show()