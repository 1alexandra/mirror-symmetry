import numpy as np
import cv2
import os


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
    img *= (int)(255 / np.max(img))
    img = img[::-1]
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    if img[0,0] == 255:
        img = 255 - img
    return img

def get_contours(path, get_all = False, min_area = 50):
    """
    input:
    image -- cv2 read bw image,
    get_all -- bool, do you need all contours (True) or only the biggest (False)
    output:
    list of comlex arrays
    """
    img = cv2.imread(path, 0)
    if img is None:
        return []
    image = binarize(img)
    w, h = image.shape
    margin = 1
    img = np.zeros((w+2*margin,h+2*margin), dtype=np.uint8)
    img[margin:-margin,margin:-margin] = image
    contours, _ = cv2.findContours(img,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    measure = np.array([cv2.contourArea(cnt) for cnt in contours])
    if get_all:
        index = np.arange(len(measure))[measure >= min_area]
    else:
        index = [np.argmax(measure)]
    return [cnt_to_u(contours[i])-margin*(1+1j) for i in index]

def from_folder(folder, get_all=True):
    return {name: get_contours(folder+'/'+name, get_all)
            for name in os.listdir(path='./'+folder)}

def fix_period(u, n_mult=2):
    """
    input:
    u -- complex array, contour points,
    n -- number of output contour points, if None, n = len(u)
    output:
    w -- complex array, contour points with equal paths along the contour
        between each adjacent pair
    """
    n = len(u) * n_mult
    u_next = np.zeros(u.shape, dtype=complex)
    u_next[:-1] = u[1:]
    u_next[-1] = u[0]
    step = np.sum(np.abs(u-u_next)) / n
    seg_ind = 0
    seg_start = u[0]
    cur_step = step
    w = []
    for i in range(n):
        w.append(seg_start)
        while True:
            seg_end = u[(seg_ind+1) % len(u)]
            seg_vec = seg_end - seg_start
            seg_len = abs(seg_vec)
            if seg_len < cur_step:
                seg_ind += 1
                seg_start = seg_end
                cur_step -= seg_len
            else:
                seg_start += seg_vec / seg_len * cur_step
                cur_step = step
                break
    return np.array(w)

def add_middles(u, mid_iters=1):
    """
    input: 
    u_ -- complex array, contour points,
    mid_iters -- int,
    output:
    preprocessed u, comlex array 'mid_iters' times added contour edge centers
    """
    u_m = []
    parts = 2 ** mid_iters
    steps = (1/parts) * np.arange(parts)
    for i in range(len(u)):
        cur = u[(i+1)%len(u)] - u[i]
        u_m += list(u[i] + steps * cur)
    return np.array(u_m)    
        
def preprocess(u):
    """
    input: 
    u_ -- complex array, contour points,
    output:
    preprocessed u, comlex array:
        max(abs(u)) = 1,
        min(re(u)) = min(im(u)) = 0;
    vec -- complex,
    scale -- double,
    u = (u-vec)/scale.
    """
    vec = np.min(np.real(u)) + np.min(np.imag(u)) * 1j
    scale = np.max(np.abs(u-vec))
    return (u-vec) / scale, vec, scale

def preprocess_inverse(u, vec, scale):
    return u * scale + vec

def index_neighbors(u, z, delta=5):
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
    u_step = np.array(list(u[1:]) + [u[0]])
    tmp = max(np.min(np.abs(u_step - u)), 1)
    delta_new = delta * tmp
    return np.arange(len(u))[np.abs(u-z) <= delta_new]
