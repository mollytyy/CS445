from skimage import draw
import numpy as np
import matplotlib.pyplot as plt

def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def specify_bottom_center(img):
    print("If it doesn't get you to the drawing mode, then rerun this function again. Also, make sure the object fill fit into the background image. Otherwise it will crash")
    fig = plt.figure()
    plt.imshow(img, cmap='gray')
    fig.set_label('Choose target bottom-center location')
    plt.axis('off')
    target_loc = np.zeros(2, dtype=int)

    def on_mouse_pressed(event):
        target_loc[0] = int(event.xdata)
        target_loc[1] = int(event.ydata)
        
    fig.canvas.mpl_connect('button_press_event', on_mouse_pressed)
    return target_loc

def align_source(object_img, mask, background_img, bottom_center):
    ys, xs = np.where(mask == 1)
    (h,w,_) = object_img.shape
    y1 = x1 = 0
    y2, x2 = h, w
    object_img2 = np.zeros(background_img.shape)
    yind = np.arange(y1,y2)
    yind2 = yind - int(max(ys)) + bottom_center[1]
    xind = np.arange(x1,x2)
    xind2 = xind - int(round(np.mean(xs))) + bottom_center[0]

    ys = ys - int(max(ys)) + bottom_center[1]
    xs = xs - int(round(np.mean(xs))) + bottom_center[0]
    mask2 = np.zeros(background_img.shape[:2], dtype=bool)
    for i in range(len(xs)):
        mask2[int(ys[i]), int(xs[i])] = True
    for i in range(len(yind)):
        for j in range(len(xind)):
            object_img2[yind2[i], xind2[j], :] = object_img[yind[i], xind[j], :]
    mask3 = np.zeros([mask2.shape[0], mask2.shape[1], 3])
    for i in range(3):
        mask3[:,:, i] = mask2
    background_img  = object_img2 * mask3 + (1-mask3) * background_img
    plt.figure()
    plt.imshow(background_img)
    return object_img2, mask2

def upper_left_background_rc(object_mask, bottom_center):
    """ 
      Returns upper-left (row,col) coordinate in background image that corresponds to (0,0) in the object image
      object_mask: foreground mask in object image
      bottom_center: bottom-center (x=col, y=row) position of foreground object in background image
    """
    ys, xs = np.where(object_mask == 1)
    (h,w) = object_mask.shape[:2]
    upper_left_row = bottom_center[1]-int(max(ys)) 
    upper_left_col = bottom_center[0] - int(round(np.mean(xs)))
    return [upper_left_row, upper_left_col]

def crop_object_img(object_img, object_mask):
    ys, xs = np.where(object_mask == 1)
    (h,w) = object_mask.shape[:2]
    x1 = min(xs)-1
    x2 = max(xs)+2
    y1 = min(ys)-1
    y2 = max(ys)+2
    object_mask = object_mask[y1:y2, x1:x2]
    object_img = object_img[y1:y2, x1:x2, :]
    return object_img, object_mask

def get_combined_img(bg_img, object_img, object_mask, bg_ul):
    combined_img = bg_img.copy()
    (nr, nc) = object_img.shape[:2]

    for b in np.arange(object_img.shape[2]):
      combined_patch = combined_img[bg_ul[0]:bg_ul[0]+nr, bg_ul[1]:bg_ul[1]+nc, b]
      combined_patch = combined_patch*(1-object_mask) + object_img[:,:,b]*object_mask
      combined_img[bg_ul[0]:bg_ul[0]+nr, bg_ul[1]:bg_ul[1]+nc, b] =  combined_patch

    return combined_img 


def specify_mask(img):
    # get mask
    print("If it doesn't get you to the drawing mode, then rerun this function again.")
    fig = plt.figure()
    fig.set_label('Draw polygon around source object')
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    xs = []
    ys = []
    clicked = []

    def on_mouse_pressed(event):
        x = event.xdata
        y = event.ydata
        xs.append(x)
        ys.append(y)
        plt.plot(x, y, 'r+')

    def onclose(event):
        clicked.append(xs)
        clicked.append(ys)
    # Create an hard reference to the callback not to be cleared by the garbage
    # collector
    fig.canvas.mpl_connect('button_press_event', on_mouse_pressed)
    fig.canvas.mpl_connect('close_event', onclose)
    return clicked

def get_mask(ys, xs, img):
    mask = poly2mask(ys, xs, img.shape[:2]).astype(int)
    fig = plt.figure()
    plt.imshow(mask, cmap='gray')
    return mask