import numpy as np
import cv2
def filter_image(im, fil):
    '''
    im: H x W floating point numpy ndarray representing image in grayscale
    fil: M x M floating point numpy ndarray representing 2D filter
    '''
    H, W = im.shape
    hs = fil.shape[0] // 2 # half of filter size
    fftsize = 1024         # should be order of 2 (for speed) and include padding
    im_fft = np.fft.fft2(im, (fftsize, fftsize))   # 1) fft im with padding
    fil_fft = np.fft.fft2(fil, (fftsize, fftsize)) # 2) fft fil, pad to same size as image
    im_fil_fft = im_fft * fil_fft;                 # 3) multiply fft images
    im_fil = np.fft.ifft2(im_fil_fft)              # 4) inverse fft2
    im_fil = im_fil[hs:hs + H, hs:hs + W]          # 5) remove padding
    im_fil = np.real(im_fil)                       # 6) extract out real part
    return im_fil
    
def fft_image(im):
    '''
    im: H x W floating point numpy ndarray representing image in grayscale
    fil: M x M floating point numpy ndarray representing 2D filter
    '''
    H, W = im.shape
    fftsize = 1024         # should be order of 2 (for speed) and include padding
    return np.fft.fft2(im, (fftsize, fftsize))

def get_mag_phase_images(im1, im2):    
    im1 = cv2.resize(im1, dsize=(im2.shape[1], im2.shape[0]))

    # compute fft, phase, mag
    im1_fft = fft_image(im1)
    im2_fft = fft_image(im2)
    im1_mag = np.abs(im1_fft)
    im1_phase = np.angle(im1_fft)
    im2_mag = np.abs(im2_fft)
    im2_phase = np.angle(im2_fft)
    return (im1,
            im1_mag,
            im1_phase,
            im2,
            im2_mag,
            im2_phase)
