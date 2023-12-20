import os
import cv2

import numpy as np
from math import floor
from numpy.linalg import svd, inv
import ffmpeg

def blendImages(sourceTransform, referenceTransform):
    '''
    Naive blending for frame stitching
    Input:
        - sourceTransform: source frame projected onto reference frame plane
        - referenceTransform: reference frame projected onto same space

    Output:
        - blendedOutput: naive blending result from frame stitching
    '''

    blendedOutput = referenceTransform
    indices = referenceTransform == 0
    blendedOutput[indices] = sourceTransform[indices]

    return (blendedOutput / blendedOutput.max() * 255).astype(np.uint8)


def get_img_corners(img):
    '''
    Returns the position of the corners of an color image
    Input:
        img: color image.
    Output:
        corners: image corners (np.array)
    '''	
    corners = np.zeros((4, 1, 2), dtype=np.float32)

    height, width, channels = img.shape
    corners[0] = (0, 0) #top left
    corners[1] = (width, 0) #top right
    corners[2] = (width, height) #bottom right
    corners[3] = (0, height) #bottom left
    
    return corners    


def video2imageFolder(input_file, output_path):
    '''
    Extracts the frames from an input video file
    and saves them as separate frames in an output directory.
    Input:
        input_file: Input video file.
        output_path: Output directorys.
    Output:
        None
    '''

    cap = cv2.VideoCapture()
    cap.open(input_file)

    if not cap.isOpened():
        print("Failed to open input video")

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    frame_idx = 0

    while frame_idx < frame_count:
        ret, frame = cap.read()

        if not ret:
            print ("Failed to get the frame {}".format(frameId))
            continue

        out_name = os.path.join(output_path, 'f{:04d}.jpg'.format(frame_idx+1))
        ret = cv2.imwrite(out_name, frame)
        if not ret:
            print ("Failed to write the frame {}".format(frame_idx))
            continue

        frame_idx += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)


def vidwrite_from_numpy(fn, images, framerate=30, vcodec='libx264'):
    ''' 
      Writes a file from a numpy array of size nimages x height x width x RGB
      # source: https://github.com/kkroening/ffmpeg-python/issues/246
    '''
    if not isinstance(images, np.ndarray):
        images = np.asarray(images)
    n,height,width,channels = images.shape
    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
            .output(fn, pix_fmt='yuv420p', vcodec=vcodec, r=framerate)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )
    for frame in images:
        process.stdin.write(
            frame
                .astype(np.uint8)
                .tobytes()
        )
    process.stdin.close()
    process.wait()

def imageFolder2mpeg(input_path, output_path='./output_video.mpeg', fps=30.0):
    '''
    Extracts the frames from an input video file
    and saves them as separate frames in an output directory.
    Input:
        input_path: Input video file.
        output_path: Output directorys.
        fps: frames per second (default: 30).
    Output:
        None
    '''

    dir_frames = input_path
    files_info = os.scandir(dir_frames)

    file_names = [f.path for f in files_info if f.name.endswith(".jpg")]
    file_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    frame_Height, frame_Width = cv2.imread(file_names[0]).shape[:2]
    resolution = (frame_Width, frame_Height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MPG1')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, resolution)

    frame_count = len(file_names)

    frame_idx = 0

    while frame_idx < frame_count:


        frame_i = cv2.imread(file_names[frame_idx])
        video_writer.write(frame_i)
        frame_idx += 1

    video_writer.release()