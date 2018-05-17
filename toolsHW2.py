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

def gauss1d(sigma, filter_length=10):
    # INPUTS
    # @ sigma         : standard deviation of gaussian distribution
    # @ filter_length : integer denoting the filter length, default is 10
    # OUTPUTS
    # @ gauss_filter  : 1D gaussian filter

    # if filter_length is even add one
    filter_length += ~filter_length % 2
    x = np.linspace(np.int(-filter_length/2),np.int(filter_length/2), filter_length)

    gauss_filter = np.exp(- (x ** 2) / (2 * (sigma ** 2)))

    gauss_filter = gauss_filter / np.sum(gauss_filter)
    return gauss_filter

def gauss2d(sigma, filter_size=10):
    # INPUTS
    # @ sigma           : standard deviation of gaussian distribution
    # @ filter_size     : integer denoting the filter size, default is 10
    # OUTPUTS
    # @ gauss2d_filter  : 2D gaussian filter

    # create a 1D gaussian filter
    gauss1d_filter = gauss1d(sigma, filter_size)[np.newaxis, :]
    # convolve it with its transpose
    
    gauss2d_filter = convolve2d(gauss1d_filter, np.transpose(gauss1d_filter))
#    gauss2d_filter = myconv2(gauss1d_filter, np.transpose(gauss1d_filter))
    return gauss2d_filter


def myconv(signal, filt): 
    # This function performs a 1D convolution between signal and filt. This
    # function should return the result of a 1D convolution of the signal and
    # the filter.
    # INPUTS
    # @ signal          : 1D image, as numpy array, of length m
    # @ filt            : 1D or 2D filter of length k
    # OUTPUTS
    # signal_filtered   : 1D filtered signal, of size (m+k-1)

    # flip the filter
    filt = np.fliplr(np.expand_dims(filt,0))

    # initialize the filtered signal
    filtered_signal = np.empty((signal.size + filt.size - 1))

    # pad original image with zeros outside of the borders
    padded_signal = np.pad(signal, ((filt.size - 1, filt.size - 1)), 'constant')

    # calculate each element of the full filtered signal
    for index in range(len(filtered_signal)):
        filtered_signal[index] = np.sum(padded_signal[index:index+filt.size] * filt)

    return filtered_signal


def myconv2(img, filt):
    # This function performs a 2D convolution between image and filt, image being a 2D image. This
    # function should return the result of a 2D convolution of these two images. 
    # INPUTS
    # @ img           : 2D image, as numpy array, size mxn
    # @ filt          : 1D or 2D filter of size kxl
    # OUTPUTS
    # img_filtered    : 2D filtered image, of size (m+k-1)x(n+l-1)

    # if filt is 1D of shape (length, None) (this is common in python), transform it to shape (1, length)
    if filt.ndim < 2:
        filt = filt[np.newaxis, :]
    if img.ndim < 2:
        img = img[np.newaxis, :]
   
    # flip the filter
    filt = np.flipud(np.fliplr(filt))
   
    # initialize the filtered image
    import pdb; pdb.set_trace()
    img_filtered = np.empty((img.shape[0] + filt.shape[0] - 1, img.shape[1] +
                             filt.shape[1] - 1))
    
    # pad original image with zeros outside of the borders
    padded_img = np.pad(img, ((filt.shape[0] - 1,), (filt.shape[1] - 1, )), 'constant')
    
    # calculate each element of the full filtered image
    for index in np.ndindex(img_filtered.shape):
        img_filtered[index] = np.sum(padded_img[index[0]:index[0]+filt.shape[0], index[1]:index[1]+filt.shape[1]] * filt)

    return img_filtered

def gconv(image, sigma, filter_size): 
    # Function that filters an image with a Gaussian filter
    # INPUTS
    # @ image         : 2D image
    # @ sigma         : the standard deviation of gaussian distribution
    # @ size          : the size of the filter
    # OUTPUTS
    # @ img_filtered  : filtered image with gaussian filter

    # Create 2D Gaussian filter
    gauss2d_filter = gauss2d(sigma, filter_size = filter_size)
    # Filter the image
    img_filtered = convolve2d(image, gauss2d_filter)
    # img_filtered   = myconv2(image, gauss2d_filter)

    return img_filtered

def DoG(img, sigma_1, sigma_2, filter_size):
    # Function that creates Difference of Gaussians (DoG) for given standard
    # deviations and filter size
    # INPUTS
    # @ img
    # @ img           : 2D image (MxN)
    # @ sigma_1       : standard deviation of of the first Gaussian distribution
    # @ sigma_2       : standard deviation of the second Gaussian distribution
    # @ filter_size   : the size of the filters
    # OUTPUTS
    # @ dog           : Difference of Gaussians of size
    #                   (M+filter_size-1)x(N_filter_size-1)

    img_f1 = gconv(img, sigma_1, filter_size)
    img_f2 = gconv(img, sigma_2, filter_size)
    dog = img_f1 - img_f2
    
    return dog 

def blur_and_downsample(img, sigma, filter_size, scale):
    # INPUTS
    # @ img                 : 2D image (MxN)
    # @ sigma               : standard deviation of the Gaussian filter to be used at
    #                         all levels
    # @ filter_size         : the size of the filters
    # @ scale               : Downscaling factor (of type float, between 0-1)
    # OUTPUTS
    # @ img_br_ds           : The blurred and downscaled image 
    
    # Blur with Gaussina filtering
    img_br = gconv(img, sigma, filter_size)
    
    # Downscale the filtered image
    img_br_ds = imresize(img_br, scale)

    return img_br_ds
       
def generate_gaussian_pyramid(img, sigma, filter_size, scale, num_levels):
    # Function that creates Gaussian Pyramid as described in the homework
    # It blurs and downsacle the iimage subsequently. Please keep in mind that
    # the first element of the pyramid is the oirignal image, which is
    # considered as the level-0. The number of levels that is given as argument
    # INCLUDES the level-0 as well. It means there will be num_levels-1 times
    # blurring and down_scaling.
    # INPUTS
    # @ img                 : 2D image (MxN)
    # @ sigma               : standard deviation of the Gaussian filter to be used at
    #                         all levels
    # @ filter_size         : the size of the filters
    # @ scale               : Downscaling factor (of type float, between 0-1)
    # OUTPUTS
    # @ gaussian_pyramid    : A list connatining images of pyramid. The first
    #                         element SHOULD be the image at the original scale
    #                         without any blurring.
    
    # Initialize the list for images in Gaussian pyramid
    # Put first the original image in the list
    gaussian_pyramid = [img]
    
    # Loop to generate filtered and downscaled images for every level
    img_given_to_next_level = img.copy()
    for l in range(num_levels-1):
        
        # Smooth and downscale using blur_and_downsample() function
        img_br_ds = blur_and_downsample(img_given_to_next_level, sigma,
                                           filter_size, scale)
        
        # Add the resulting image to the list
        gaussian_pyramid.append(img_br_ds)
        
        # Continue with the current filtered image
        img_given_to_next_level = img_br_ds.copy()
    
    return gaussian_pyramid
