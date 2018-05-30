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
import cv2
from toolsHW2 import *
from toolsHW4 import *
from skimage.feature import match_template

##############################################################################
#                           Main script starts here                          #
##############################################################################


# ##################
# Select the Dataset to Calculate 1:Dataset a  0:Dataset b
# ##################
# dataset = 1
for dataset in range(2):
    # ##################
    # Store the Detected Eye in an Image 1:Yes 0:No
    # ##################
    saveEyeTexture = 1

    if dataset == 1:
        folder = 'project_data/a/'
        folderEye = 'project_data/a_eye/'
        newfolder = 'solution_a/'
        solutionFilefolder = ''
        solutionFileName = 'solution_a'
        xFirstSolution, yFirstSolution = 348, 191
        eye_detect_threshold = 0.4
        use_DoG = 1
        use_sharpener = 1
        use_contrast_correction = 1
        use_gaussfilter_around_solution = 1
    else:
        folder = 'project_data/b/'
        folderEye = 'project_data/b_eye/'
        newfolder = 'solution_b/'
        solutionFilefolder = ''
        solutionFileName = 'solution_b'
        xFirstSolution, yFirstSolution = 439, 272
        eye_detect_threshold = 0.2
        use_DoG = 0
        use_sharpener = 1
        use_contrast_correction = 1
        use_gaussfilter_around_solution = 1

    os.makedirs(folderEye, exist_ok=True)
    os.makedirs(newfolder, exist_ok=True)

    xySolutions = []
    corrNew = []

    model_m, model_c = None, None
    ransac_iterations = 100
    ransac_threshold = 2
    n_samples = 2

    ratio = 0
    i = -4

    previous_m, previous_c = 0, 0
    xEyeCenter, yEyeCenter, xEyeSize = xFirstSolution,yFirstSolution,0
    previous_xEyeCenter, previous_yEyeCenter, previous_xEyeSize = xEyeCenter, yEyeCenter, xEyeSize
    first_xEyeSize = None

    eye_texture_origImg = None
    eye_texture_img = None
    previous_eye_texture_img = None
    previous_texture_img = None
    texture_patch = None
    previous_texture_patch = None
    previous_texture_patch_gen2 = None
    previous_texture_patch_gen3 = None
    first_texture_patch  = None
    corrdinateChange = tuple((0,0))
    locNew = tuple((1,1))
    locFirst = tuple((1,1))
    locGen1 = tuple((1,1))
    locGen2 = tuple((1,1))
    locGen3 = tuple((1,1))
    averageLoc = tuple((1,1))
    solutionInEye = tuple((1,1))

    sigma_1 = 1
    sigma_2 = 3
    filter_size = 30

    dominant_channel = 0

    xSolution, ySolution = xFirstSolution , yFirstSolution
    previous_xSolution, previous_ySolution = xFirstSolution, yFirstSolution

    # Load the images and sort the paths to the sequenced names
    filenames = imagesfilename_from_folder(folder)
    filenames = np.sort(filenames)

    # timestamp start
    ts1 = time.time()
    st = datetime.datetime.fromtimestamp(ts1).strftime('%Y-%m-%d %H:%M:%S')
    print(st)

    for index, filename in enumerate(filenames): # loop through all images in folder
        filename = filenames[index]
        print(filename)
        name, filetype = filename[:i], filename[i:]
        newfilename = str(name)+"_solution"+str(filetype)

        filepath = os.path.join(folder,filename)
        newfilepath = os.path.join(newfolder, newfilename)
        eyefilepath = os.path.join(folderEye, filename)

        origImage = plt.imread(filepath)
        hist, bins = np.histogram(origImage.ravel(), 256, [0, 256], density=True)

        # ##################
        # Color or Grayscale
        # ##################
        previous_dominant_channel = dominant_channel
        dominant_channel = find_dominant_channel(origImage)[0] # set to -1 if you want color approach

        if previous_dominant_channel != dominant_channel:
            print("Dominant_channel changed from "+str(previous_dominant_channel)+" to "+str(dominant_channel))

        if dominant_channel > -1:
            image = origImage[:, :, dominant_channel]
        else:
            image = origImage
        # end Color or Grayscale
        # ##################

        # ##################
        # Detect EyeCenter And Size
        # ##################
        previous_xEyeCenter, previous_yEyeCenter,previous_xEyeSize = xEyeCenter, yEyeCenter, xEyeSize
        xEyeCenter, yEyeCenter, xEyeSize = detecingEyeCenterAndSize(image,eye_detect_threshold) # edges
        newEyeCenterDifference = tuple((int(yEyeCenter - previous_yEyeCenter), int(xEyeCenter - previous_xEyeCenter)))
        print("EyeCenter X:"+str(xEyeCenter)+" Y:"+str(yEyeCenter)+"Previous Diff:"+str(newEyeCenterDifference[1])+" Y:"+str(newEyeCenterDifference[0]))

        if first_xEyeSize is None:
            first_xEyeSize = xEyeSize

        # end Detect EyeCenter And Size
        # ##################

        # ##################
        # tried calculate RANSAC in a first attempt
        # ##################
        # model_m, model_c = performRANSAC(edge_pts, edge_pts_xy, ransac_iterations, ransac_threshold, n_samples, ratio)
        #
        # m = model_m
        # c = model_c
        # end calculate RANSAC
        # ##################

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

        sys.stdout.write('\r')
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        print(st)

        f2 = plt.figure()

        # ##################
        # show the current image
        # ##################
        # plt.imshow(image)
        # end the current image
        # ##################

        first_EyeRadius = first_xEyeSize
        square = int(first_xEyeSize / 2) + 26
        h, w = square, square
        eye_gaussfilter = gauss2d(square, filter_size= first_EyeRadius * 2 + 1)
        gaussfilter = gauss2d(square, filter_size= square * 2 + 1)

        nonGaussfilter = np.zeros((square * 2 + 1, square * 2 + 1))
        nonGaussfilter[nonGaussfilter == 0] = 1

        # ##################
        # pick a texture_img around the previous xySolution
        # ##################
        if dominant_channel > -1: # greyscale approach
            texture_img = image[ySolution - h:ySolution + h + 1,xSolution - w :xSolution + w + 1]

            eye_texture_origImg = origImage[yEyeCenter - first_EyeRadius:yEyeCenter + first_EyeRadius + 1,
                              xEyeCenter - first_EyeRadius:xEyeCenter + first_EyeRadius + 1]

            eye_texture_img = image[yEyeCenter - first_EyeRadius  :yEyeCenter + first_EyeRadius  + 1,
                          xEyeCenter - first_EyeRadius :xEyeCenter + first_EyeRadius  + 1]
            solutionInEye = tuple((first_EyeRadius + (ySolution - yEyeCenter),first_EyeRadius + (xSolution - xEyeCenter)))

            # ##################
            # create eye_gaussfilter around the ≈ xySolution
            if use_gaussfilter_around_solution == 1:
                eye_gaussfilter = np.zeros((first_EyeRadius * 2 + 1, first_EyeRadius * 2 + 1))
                eye_gaussfilter[eye_gaussfilter == 0] = gaussfilter[0][0] # set at least the value of the lowest gaussfilter

                filterShape = np.shape(gaussfilter)
                eyefilterShape = np.shape(eye_gaussfilter)

                maxRangeX = filterShape[1]
                maxRangeY = filterShape[0]

                startIndexX = w
                startIndexY = h

                usedGaussfilter = nonGaussfilter # gaussfilter

                if solutionInEye[0] +  h + 1 > eyefilterShape[0]:
                    startIndexY = (solutionInEye[0] +  h + 1 - eyefilterShape[0]) + h
                    usedGaussfilter = nonGaussfilter

                if solutionInEye[1] + w + 1 > eyefilterShape[1]:
                    startIndexX = (solutionInEye[1] +  w + 1 - eyefilterShape[1]) + w
                    usedGaussfilter = nonGaussfilter

                for indexX in range(maxRangeX):
                    for indexY in range(maxRangeY):
                        eye_gaussfilter[solutionInEye[0] - startIndexY + indexY][solutionInEye[1] - startIndexX + indexX] = gaussfilter[indexX][indexY]

            # end create eye_gaussfilter around the previous xySolution
            # ##################

            if use_sharpener == 1:
                sharp = np.asarray([0, 0, 0, 0, 4, 0, 0, 0, 0]).reshape((1, 9))
                eye_texture_img = convolve2d(eye_texture_img, sharp, mode='same', boundary='fill', fillvalue=0)

            if use_contrast_correction == 1:
                min = np.min(eye_texture_img)
                max = np.max(eye_texture_img)
                eye_texture_imgShape = np.shape(eye_texture_img)

                for indexX in range(eye_texture_imgShape[1]):
                  for indexY in range(eye_texture_imgShape[0]):
                    eye_texture_img[indexY][indexX] = (eye_texture_img[indexY][indexX] - min) / (max - min) * 255

            if use_DoG == 1:
                eye_texture_img = DoG(eye_texture_img, sigma_1, sigma_2, filter_size, "same")

            eye_texture_img = eye_texture_img * eye_gaussfilter

            texture_img = eye_texture_img[solutionInEye[0] - h:solutionInEye[0] + h + 1, solutionInEye[1] - w:solutionInEye[1] + w + 1]

            texture_filtered = texture_img # * gaussfilter
        else :
            # color approach not working anymore at the moment
            texture_img = image[ySolution - h:ySolution + h,xSolution - w:xSolution + w, :]
            texture_filtered = texture_img * gaussfilter
            # pick a texture_img around the previous xySolution
            ######

        #####
        # default eye_texture_img
        if previous_eye_texture_img is None:
            previous_eye_texture_img = eye_texture_img.copy()
        # end default eye_texture_img
        ######

        # ##################
        # tried create a time_difference_eye_texture_img out of convolutions
        # img_f1 = gconv(eye_texture_img, sigma_1, filter_size, mode='same')
        # img_f2 = gconv(previous_eye_texture_img, sigma_2, filter_size, mode='same')
        # time_difference_eye_texture_img = (img_f1 - img_f2)

        # texture_filtered = image[solutionInEye[0] - h:solutionInEye[0] + h + 1,solutionInEye[1] - w:solutionInEye[1] + w + 1]


        #####
        # pick the matching texture_patch
        if previous_texture_img is None:
            previous_texture_img = texture_filtered.copy()
            entrypoint = int(square / 2)

        enlargePatchByExtraPixel = 0 # 5
        new_texture_patch = texture_filtered[entrypoint - enlargePatchByExtraPixel:entrypoint + h + enlargePatchByExtraPixel,entrypoint - enlargePatchByExtraPixel :entrypoint + w + enlargePatchByExtraPixel] # * -1 # multiply by -1 to invert the patch as the time_difference_texture_img is inverted
        # end pick the maching texture_patch
        ######

        # ##################
        # create time_difference_texture_img as differenc from this to previous texture_img around the solution
        # ##################

        # ##################
        # tried to useTimeDifference Patch
        # useTimeDifferenceInPatch = 0
        # if useTimeDifferenceInPatch == 1:
        #     time_difference_texture_img = texture_filtered - previous_texture_img
        # else:
        #     time_difference_texture_img = texture_filtered * -1 # invert because without timmedifference in Patch and time difference in eyetexture it is not inverted
        # end tried to useTimeDifference Patch
        # ##################

        # ##################
        #  tried to min out the extrem lights (limiter) is not working well at the moment
        #min = np.min(eye_texture_img)
        #max = np.max(eye_texture_img)
        #eye_texture_img[eye_texture_img == max] = (min + max) / 2
        # end tried to min out the extrem lights
        # ##################

        # ##################
        # tried calculate RANSAC only on the detected Eye
        # calculate_RANSAC_onEye(eye_texture_img)

        ######
        # store detected Eye with last Solution
        if saveEyeTexture == 1:
            # ##################
            # tried mirroring effects to better detect
            # eye_texture_img = mirrorTextureAt45Degree(eye_texture_img, dominant_channel = dominant_channel)

            # ##################
            # tried DoG on Eye to better detect
            # plt.imshow(DoG(eye_texture_img,sigma_1,sigma_2,filter_size,"same"))

            plt.imshow(eye_texture_origImg) # eye_texture_img black and white
            # plt.scatter(x=[solutionInEye[1]], y=[solutionInEye[0]], c='r', s=10)  # show the average of the found matches
            f2.show()
            f2.savefig(eyefilepath, dpi=90, bbox_inches='tight')

        time_difference_eye_texture_img = eye_texture_img * eye_gaussfilter # * -1 # (eye_texture_img - previous_eye_texture_img) * eye_gaussfilter

        # end create time_difference_texture_img as differenc from this to previous texture_img around the solution
        # ##################

        # ##################
        # match the texture_patch to the selected time_difference_texture_img
        # ##################
        if index > 0:  # is not first image

            previous_xSolution, previous_ySolution = xSolution, ySolution

            # ##################
            # calculate of the patch finally used for this iteration
            # ##################

            ######
            # tryied to calculate the difference between new and first patch
            # corrdinateChangeToFirst = match_texture_patch(new_texture_patch, first_texture_patch)
            # print("corrdinateChangeToFirst: " + str(corrdinateChangeToFirst))

            ######
            # tryied to calculate the difference between new and previous patch
            # corrdinateChange = match_texture_patch(new_texture_patch, previous_texture_patch)
            # print("corrdinateChange: " + str(corrdinateChange))

            ######
            # tryied to use different variation of new and old patches as the finally used for this or next iteration
            texture_patch = new_texture_patch # previous_texture_patch # new_texture_patch # (new_texture_patch + previous_texture_patch) / 2
            # end calculate of the patch finally used for this iteration
            ######

            # corrNew = match_template(time_difference_eye_texture_img, texture_patch, pad_input=True)
            # corrNewMax = np.max(corrNew)
            # locNew = tuple((np.where(corrNew == np.max(corrNew))[0][0], np.where(corrNew == np.max(corrNew))[1][0]))

            ######
            # show found solution
            #plt.imshow(time_difference_eye_texture_img)
            #plt.imshow(texture_patch)
            #plt.scatter(x=[locNew[1]], y=[locNew[0]], c='g', s=10)
            #plt.show()

            corrGen1 = match_template(time_difference_eye_texture_img, previous_texture_patch, pad_input=True)
            locGen1 = tuple((np.where(corrGen1 == np.max(corrGen1))[0][0], np.where(corrGen1 == np.max(corrGen1))[1][0]))
            #corrGen1Max = np.max(corrGen1)

            corrGen2 = match_template(time_difference_eye_texture_img, previous_texture_patch_gen2, pad_input=True)
            locGen2 = tuple((np.where(corrGen2 == np.max(corrGen2))[0][0], np.where(corrGen2 == np.max(corrGen2))[1][0]))
            diffToGen2 = tuple((locGen2[0] - locGen1[0], locGen2[1] - locGen1[1]))
            #corrGen2Max = np.max(corrGen2)

            corrGen3 = match_template(time_difference_eye_texture_img, previous_texture_patch_gen3, pad_input=True)
            locGen3 = tuple((np.where(corrGen3 == np.max(corrGen3))[0][0], np.where(corrGen3 == np.max(corrGen3))[1][0]))
            #corrGen3Max = np.max(corrGen3)
            diffToGen3 = tuple((locGen3[0] - locGen1[0], locGen3[1] - locGen1[1]))

            corrFirst = match_template(time_difference_eye_texture_img, first_texture_patch, pad_input=True)
            locFirst = tuple((np.where(corrFirst == np.max(corrFirst))[0][0], np.where(corrFirst == np.max(corrFirst))[1][0]))
            #corrFirstMax = np.max(corrFirst)
            diffToFirst = tuple((locFirst[0] - locGen1[0], locFirst[1] - locGen1[1]))

            ######
            # tryied to use only the best match from last 3 gen
            #corrNew = [corrNewMax,corrGen1Max,corrGen2Max,corrGen3Max,corrFirstMax]
            #print("diffToGen2: " + str(diffToGen2) + " diffToGen3: " + str(diffToGen3) + " diffToFirst: " + str(diffToFirst))

            #n = np.where(corrNew == np.max(corrNew))
            #print("n: " + str(n) + "corrNew: " + str(np.max(corrNew)) + "corrNewMax Pos: " + str(np.where(corrNew == np.max(corrNew))))

            averageLoc = locGen1 # locNew
            #if n == 0: # locNew
            #    averageLoc = tuple((np.where(corrNew == np.max(corrNew))[0][0], np.where(corrNew == np.max(corrNew))[1][0]))
            #elif n == 1: # locGen1
            #    averageLoc = tuple((np.where(corrGen1 == np.max(corrGen1))[0][0], np.where(corrGen1 == np.max(corrGen1))[1][0]))
            #elif n == 2: # locGen2
            #    averageLoc = tuple((np.where(corrGen2 == np.max(corrGen2))[0][0], np.where(corrGen2 == np.max(corrGen2))[1][0]))
            #elif n == 3: # locGen3
            #    averageLoc = tuple((np.where(corrGen3 == np.max(corrGen3))[0][0], np.where(corrGen3 == np.max(corrGen3))[1][0]))
            #else: # locFirst
            #    averageLoc = tuple((np.where(corrFirst == np.max(corrFirst))[0][0], np.where(corrFirst == np.max(corrFirst))[1][0]))

            ######
            # tryied to use different variation of matches as final solution for this iteration
            # averageLoc = locNew
            # averageLoc = tuple((int((locNew[0] + locGen1[0] + locGen2[0] + locGen3[0])/4), int((locNew[1] + locGen1[1] + locGen2[1] + locGen3[1])/4)))

            #print("corrdinateChangeMatch: " + str(locNew) + " locGen1: " + str(locGen1) + " locGen2: " + str(locGen2) + " locGen3: " + str(locGen3) + " locFirst: " + str(locFirst) + " averageLoc: " + str(averageLoc))

            ######
            # tryied to calculate the difference between new average and previous patch
            xFirstSolutionCandidate, yFirstSolutionCandidate = xEyeCenter - first_EyeRadius + averageLoc[1], yEyeCenter - first_EyeRadius + averageLoc[0]

            texture_filtered_SolutionCandidate = image[yFirstSolutionCandidate - h:yFirstSolutionCandidate + h + 1,
                          xFirstSolutionCandidate - w:xFirstSolutionCandidate + w + 1] # * gaussfilter * -1

            ######
            # tryied to use two gaussconvolutions to create a final texture
            #img_f1 = gconv(texture_filtered, sigma_1, filter_size)
            #img_f2 = gconv(texture_filtered_SolutionCandidate, sigma_2, filter_size)
            #texture_filtered_SolutionCandidate = img_f1 - img_f2

            ######
            # tryied to match texture_filtered_SolutionCandidate with the last 3 gen and take the average of the best two of three
            corrdinateChangeGen1 = match_texture_patch(previous_texture_patch, texture_filtered_SolutionCandidate )

            corrdinateChangeGen2 = match_texture_patch(previous_texture_patch_gen2, texture_filtered_SolutionCandidate )
            diffToGen2 = np.sqrt((corrdinateChangeGen2[0] - corrdinateChangeGen1[0]) ** 2 + (corrdinateChangeGen2[1] - corrdinateChangeGen1[1]) ** 2)

            corrdinateChangeGen3 = match_texture_patch(previous_texture_patch_gen3, texture_filtered_SolutionCandidate)
            diffToGen3 = np.sqrt((corrdinateChangeGen3[0] - corrdinateChangeGen1[0]) ** 2 + (corrdinateChangeGen3[1] - corrdinateChangeGen1[1]) ** 2)

            corrdinateChangeFirst = match_texture_patch(first_texture_patch, texture_filtered_SolutionCandidate )
            diffToFirst = np.sqrt((corrdinateChangeFirst[0] - corrdinateChangeGen1[0]) ** 2 + (corrdinateChangeFirst[1] - corrdinateChangeGen1[1]) ** 2)

            ######
            # take the average of the best two of three
            diffTo = [diffToGen2,diffToGen3,diffToFirst]
            n = np.where(diffTo == np.max(diffTo))[0][0]
            # print( ""+str(n)+"diffToGen2: " + str(diffToGen2) + " diffToGen3: " + str(diffToGen3) + " diffToFirst: " + str(diffToFirst))
            if n == 0: # diffToGen2 is max take average of point diffToGen3 and diffToFirst
                finalCorrdinateChangeY = int(((corrdinateChangeGen3[0] - corrdinateChangeGen1[0]) + (corrdinateChangeFirst[0] - corrdinateChangeGen1[0]))/2)
                finalCorrdinateChangeX = int(((corrdinateChangeGen3[1] - corrdinateChangeGen1[1]) + (corrdinateChangeFirst[1] - corrdinateChangeGen1[1]))/2)
            elif n == 1: # diffToGen3 is max take average of point diffToGen2 and diffToFirst
                finalCorrdinateChangeY = int(((corrdinateChangeGen2[0] - corrdinateChangeGen1[0]) + (corrdinateChangeFirst[0] - corrdinateChangeGen1[0]))/2)
                finalCorrdinateChangeX = int(((corrdinateChangeGen2[1] - corrdinateChangeGen1[1]) + (corrdinateChangeFirst[1] - corrdinateChangeGen1[1]))/2)
            elif n == 2: # diffToFirst is max take average of point diffToGen2 and diffToGen3
                finalCorrdinateChangeY = int(((corrdinateChangeGen2[0] - corrdinateChangeGen1[0]) + (corrdinateChangeGen3[0] - corrdinateChangeGen1[0]))/2)
                finalCorrdinateChangeX = int(((corrdinateChangeGen2[1] - corrdinateChangeGen1[1]) + (corrdinateChangeGen3[1] - corrdinateChangeGen1[1]))/2)

            finalCorrdinateChange = tuple((finalCorrdinateChangeY,finalCorrdinateChangeX))
            print("finalCorrdinateChange: " + str(finalCorrdinateChange))

            # show found solution
            # plt.imshow(texture_filtered_SolutionCandidate)
            # plt.scatter(x=[loc[1]], y=[loc[0]], c='g', s=10)
            # plt.show()

            xSolution, ySolution = xEyeCenter - first_EyeRadius + averageLoc[1], yEyeCenter - first_EyeRadius + averageLoc[0]

            xSolution = xSolution + finalCorrdinateChange[1]
            ySolution = ySolution + finalCorrdinateChange[0]

            ######
            # store previous patches
            previous_texture_patch_gen3 = previous_texture_patch_gen2
            previous_texture_patch_gen2 = previous_texture_patch
            previous_texture_patch = new_texture_patch
            # end store previous patches
            #######

        else: # is first image set sliding window the ground variables
            texture_patch = new_texture_patch
            first_texture_patch = new_texture_patch
            previous_texture_patch_gen3 = new_texture_patch
            previous_texture_patch_gen2 = new_texture_patch
            previous_texture_patch = new_texture_patch

        # end match the texture_patch to the selected time_difference_texture_img
        # ##################


        plt.imshow(origImage) # show original image
        # plt.imshow(image) # show image used for calculation
        # plt.imshow(eye_gaussfilter) # show the used eye_gaussfilter for this iteration
        # plt.imshow(gaussfilter) # show the used gaussfilter on the patch for this iteration
        plt.imshow(time_difference_eye_texture_img) # show the eye texture used for match in this iteration
        plt.imshow(texture_patch) # show the patch used for match in this iteration

        plt.scatter(x=[solutionInEye[1]], y=[solutionInEye[0]], c='g', s=10)  # show the previous solution

        # plt.scatter(x=[locNew[1]], y=[locNew[0]], c='b', s=10) # show the 1. found match
        plt.scatter(x=[locGen2[1]], y=[locGen2[0]], c='#EDB205', s=10)  # show the 1. found match
        plt.scatter(x=[locGen3[1]], y=[locGen3[0]], c='#06DEE4', s=10)  # show the 1. found match
        plt.scatter(x=[locFirst[1]], y=[locFirst[0]], c='b', s=10) # show the average of the found matches

        plt.scatter(x=[locGen1[1]], y=[locGen1[0]], c='#9929BD', s=10)  # show the 1. found match

        # xLoc = averageLoc[1] - corrdinateChange[1]
        # yLoc = averageLoc[0] - corrdinateChange[0]
        plt.scatter(x=[xSolution], y=[ySolution], c='r', s=10) # show the average of the found matches



        # ##################
        # paint a red square arount solution coordinate (patch area used in the next iteration)
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
                facecolor= 'red', #
                alpha=0.2
            )
        )
        # end paint a red square arount solution coordinate
        # ##################

        # ##################
        # paint a greed square arount solution coordinate (focus area where the gaus will be applied in the next iteration)
        # ##################
        n_plots = 1
        ax1 = plt.subplot(n_plots, n_plots, 1)
        ax1.add_patch(
            plt.Rectangle(
                (previous_xSolution - w, previous_ySolution - h),  # (x,y)
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
                       facecolor= None, #"#00ccff",
                       alpha=0.3
                       )
        )
        plt.plot([xSolution], [ySolution], marker='o', markersize=5, color="red")
        # end paint a blue circle arount the area recognized as eye
        # ##################

        # ##################
        # paint a blue circle arount the area previously recognized as eye
        # ##################
        # ax1.add_patch(
        #     plt.Circle((previous_xEyeCenter, previous_yEyeCenter), radius=previous_xEyeSize,
        #                edgecolor='red',
        #                facecolor="#cccccc",
        #                alpha=0.3
        #                )
        # )
        # plt.plot([xSolution], [ySolution], marker='o', markersize=5, color="red")
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
        previous_eye_texture_img = eye_texture_img.copy()

    b = open(solutionFilefolder+solutionFileName+'.csv', 'w')
    a = csv.writer(b)
    a.writerows(xySolutions)
    b.close()

#plt.axis('off')
#plt.show()