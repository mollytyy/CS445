# -*- coding: utf-8 -*-
""" Contains displaying of image / hdr images """

# imports
import cv2
import numpy as np
import matplotlib.pyplot as plt



def rescale_images_linear(le):
    '''
    Helper function to rescale images in visible range
    '''
    le_min = le[le != -float('inf')].min()
    le_max = le[le != float('inf')].max()
    le[le==float('inf')] = le_max
    le[le==-float('inf')] = le_min

    le = (le - le_min) / (le_max - le_min)

    return le

def display_images_linear_rescale(images):
    '''
    Given N images, display in a row after rescaling

    Args:
      - images: NxHxWxC float32 ndarray image
    '''
    N, H, W, C = images.shape
    fix, axes = plt.subplots(1, N, figsize=(15,15))
    [a.axis('off') for a in axes.ravel()]
    rescaled_images = rescale_images_linear(images)
    for n in range(N):
        axes[n].imshow(rescaled_images[n])
