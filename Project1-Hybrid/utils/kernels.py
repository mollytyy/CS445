import numpy as np
from scipy import signal


def gaussian_kernel2d(kernel_size=21, std=3):
    """ Returns a 2D Gaussian kernel array. """
    gaussian_kernel1d = signal.gaussian(kernel_size, std=std)[:, np.newaxis]
    kernel = np.outer(gaussian_kernel1d, gaussian_kernel1d)
    return kernel / kernel.sum()


def box_kernel2d(kernel_size=21, box_radius=5):
    """ Returns 2D Box kernel arrau"""
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel_center = kernel_size // 2
    kernel[kernel_center-box_radius:kernel_center+box_radius + 1, 
           kernel_center-box_radius:kernel_center+box_radius + 1] = (2 * box_radius + 1) ** -2
    return kernel

def sobel_kernel2d():
    kernel = np.zeros((3, 3), dtype=np.float32)
    kernel[0, 0] = 1
    kernel[1, 0] = 2
    kernel[2, 0] = 1
    kernel[0, 2] = -1
    kernel[1, 2] = -2
    kernel[2, 2] = -1
    return kernel

def log_kernel2d(kernel_size=21, std=3):
    gaussian = gaussian_kernel2d(kernel_size, std)
    kernel = np.zeros_like(gaussian)
    kernel[kernel_size // 2, kernel_size // 2] = 1
    kernel -= gaussian
    return kernel
