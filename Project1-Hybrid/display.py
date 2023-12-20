import numpy as np
import matplotlib.pyplot as plt
import sys
from .image_processing import filter_image
from .kernels import gaussian_kernel2d

def _to_log_amplitude(frequency_image):    
    shifted_image = np.fft.fftshift(frequency_image)
    amplitude_image = np.abs(shifted_image + sys.float_info.epsilon)
    log_amplitude_image = np.log(amplitude_image)
    return log_amplitude_image

def display_intensity_image(intensity_image):
    fig = plt.figure(figsize=(15, 10))
    plt.title('Intensity Image')
    plt.imshow(intensity_image, cmap='gray')
    plt.axis('off')
    plt.show()
    
def display_frequency_image(frequency_image):
    '''
    frequency_image: H x W floating point numpy ndarray representing image after FFT
                     in grayscale
    '''
    log_amplitude_image = _to_log_amplitude(frequency_image)
    log_amplitude_image = filter_image(log_amplitude_image, gaussian_kernel2d(3, 0.5))
    fig = plt.figure(figsize=(15, 10))
    sv = np.sort(log_amplitude_image.flatten())
    
    vmin = sv[int(round(len(sv) * 0.02))]
    vmax = sv[int(round(len(sv) * 0.98))]
    plt.imshow(log_amplitude_image, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.axis('off')
    plt.show()
    
def display_intensity_and_frequency_images(intensity_image, frequency_image):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    log_amplitude_image = _to_log_amplitude(frequency_image)
    log_amplitude_image = filter_image(log_amplitude_image, gaussian_kernel2d(3, 0.5))
    sv = np.sort(log_amplitude_image.flatten())

    vmin = sv[int(round(len(sv) * 0.02))]
    vmax = sv[int(round(len(sv) * 0.98))]
    
    axes[0].set_title('Intensity Image')
    axes[1].set_title('Log Magnitude Image')
    axes[0].imshow(intensity_image, cmap='gray')
    cax = axes[1].imshow(log_amplitude_image, vmin=vmin, vmax=vmax)
    fig.colorbar(cax)
    axes[0].axis('off')
    axes[1].axis('off')
    plt.show()
        


def display_filtering_process(image, kernel):
    H, W = image.shape
    hs = kernel.shape[0] // 2 
    fftsize = 1024
    fft_intensity_image = np.fft.fft2(image, (fftsize, fftsize))
    fft_kernel = np.fft.fft2(kernel, (fftsize, fftsize))
    fft_filtered_image = fft_intensity_image * fft_kernel
    filtered_image = np.fft.ifft2(fft_filtered_image)
    filtered_image = filtered_image[hs:hs + H, hs:hs + W]
    filtered_image = np.real(filtered_image)
    
    
    # setup for drawing
    intensity_image = image
    filter_image = np.pad(kernel, 1, 'constant', constant_values=(0))
    fft_intensity_log_amplitude = _to_log_amplitude(fft_intensity_image)
    fft_filtered_log_amplitude = _to_log_amplitude(fft_filtered_image)
    fft_kernel_amplitude = np.abs(np.fft.fftshift(fft_kernel) + sys.float_info.epsilon)
    
    
    sv = np.sort(fft_intensity_log_amplitude.flatten())
    
    vmin = sv[int(round(len(sv) * 0.005))]
    vmax = sv[-1]
    
    # actually draw
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    [a.axis('off') for a in axes.ravel()]
    axes[0, 0].set_title('Intensity Image')
    axes[0, 0].imshow(intensity_image, cmap='gray')

    axes[0, 1].set_title('Filter')
    axes[0, 1].imshow(filter_image, cmap='gray')
    
    axes[0, 2].set_title('Filtered Image')
    axes[0, 2].imshow(filtered_image, cmap='gray')
    
    axes[1, 0].set_title('Log Magnitude of FFT Transformed Intensity Image')
    ax10 = axes[1, 0].imshow(fft_intensity_log_amplitude, vmin=vmin, vmax=vmax)
    fig.colorbar(ax10, ax=axes[1, 0])
    
    axes[1, 1].set_title('Filter FFT')
    ax11 = axes[1, 1].imshow(fft_kernel_amplitude)
    fig.colorbar(ax11, ax=axes[1, 1])
    
    axes[1, 2].set_title('Log Magnitude of FFT Transformed Filtered Image')
    ax12 = axes[1, 2].imshow(fft_filtered_log_amplitude, vmin=vmin, vmax=vmax)
    fig.colorbar(ax12, ax=axes[1, 2])
    
    
    color_bar_targets = [ax10, ax11, ax12]
    
    plt.show()
    return (intensity_image,
            filter_image,
            filtered_image,
            fft_intensity_log_amplitude,
            fft_kernel_amplitude,
            fft_filtered_log_amplitude)

def display_mag_phase_images(im1, im1_mag, im1_phase, im2, im2_mag, im2_phase):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    
    im1_log_mag = _to_log_amplitude(im1_mag)
    im1_shifted_phase = np.fft.fftshift(im1_phase)
    im2_log_mag = _to_log_amplitude(im2_mag)
    im2_shifted_phase = np.fft.fftshift(im2_phase)
    
    [a.axis('off') for a in axes.ravel()]
    axes[0, 0].set_title('Image 1 Intensity')
    axes[0, 0].imshow(im1, cmap='gray')

    axes[0, 1].set_title('Image 1 FFT Magnitude')
    axes[0, 1].imshow(im1_log_mag)
    axes[0, 2].set_title('Image 1 FFT Phase')
    axes[0, 2].imshow(im1_shifted_phase)

    axes[1, 0].set_title('Image 2 Intensity')
    axes[1, 0].imshow(im2, cmap='gray')

    axes[1, 1].set_title('Image 2 FFT Magnitude')
    axes[1, 1].imshow(im2_log_mag)
    axes[1, 2].set_title('Image 2 FFT Phase')
    axes[1, 2].imshow(im2_shifted_phase)
    
def merge_and_display_mag_phase(img, mag, phase):
    merged = mag * (np.cos(phase) + 1j * np.sin(phase))
    merged_image = np.fft.ifft2(merged)
    real_merged_image = np.real(merged_image)[:img.shape[0], :img.shape[1]]
    plt.figure(figsize=(15, 10))
    plt.axis('off')
    plt.imshow(real_merged_image, cmap='gray')
    return real_merged_image