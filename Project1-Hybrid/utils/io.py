import cv2
import numpy as np

def read_image(image_path):
    intensity_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return intensity_image.astype(np.float32) / 255