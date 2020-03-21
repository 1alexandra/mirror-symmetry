import numpy as np
import cv2

def u_to_cnt(u):
    """
    input: complex array
    output: cv2 contour (int32)
    """
    return np.array([[np.real(u),
                      np.imag(u)
                     ]]).T.reshape((-1,1,2)).astype(np.int32)

def cnt_to_u(contour):
    """
    input: cv2 contour
    output: complex array
    """
    cnt = contour.reshape((-1,2))
    return cnt[:,0] + 1j * cnt[:,1]

def binarize(image):
    """
    input:
    image -- cv2 read bw image,
    output:
    image -- cv2 binarized image, with only 0 and 255 values.
    """
    blurred = cv2.GaussianBlur(image, (5,5), 0)
    img = np.array(blurred, dtype=np.uint8)
    if blurred[0,0] == 255:
        img = 255 - img
    img *= (int)(255 / np.max(img))
    if len(np.unique(blurred)) != 2:
        img = cv2.adaptiveThreshold(img, 255, 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 
                                    5, 0)
    return img
    
def get_contours(image, get_all = False):
    """
    input:
    image -- cv2 read bw image,
    get_all -- bool, do you need all contours (True) or only the biggest (False)
    output:
    list of comlex arrays
    """
    img = binarize(image)
    contours, _ = cv2.findContours(img,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not get_all:
        measure = [cv2.contourArea(cnt) for cnt in contours]
        contours = [contours[np.argmax(measure)]]
    return [cnt_to_u(cnt) for cnt in contours]

def preprocess(u, mid_iters = 1):
    """
    input: 
    u -- complex array, contour points,
    mid_iters -- int,
    output:
    preprocessed u, comlex array:
        max(abs(u)) = 1,
        min(re(u)) = min(im(u)) = 0;
        'mid_iters' times added contour edge centers,
    vec -- complex,
    scale -- double,
    u = (u-vec)/scale.
    """
    vec = np.min(np.real(u)) + np.min(np.imag(u)) * 1j
    scale = np.max(np.abs(u))
    u -= vec
    u /= scale
    u_m = []
    parts = 2 ** mid_iters
    steps = (1/parts) * np.arange(parts)
    for i in range(len(u)):
        vec = u[(i+1)%len(u)] - u[i]
        u_m += list(u[i] + steps * vec)
    return np.array(u_m), vec, scale

def preprocess_inverse(u, vec, scale):
    return u * scale + vec

def index_neighbors(u, z, delta = 5):
    """
    input:
    u -- complex array, contour points,
    z -- complex point,
    delta -- double, scaling factor of neighbourhood.
    output:
    indexes of u elements which are nearest neighbours of points z.
    """
    if delta is None:
        return np.arange(len(u))
    u_step = np.array(list(u[1:])+[u[0]])
    tmp = max(np.min(np.abs(u_step-u)),1)
    delta_new = delta * tmp ### было /
    return np.arange(len(u))[np.abs(u-z) <= delta_new]