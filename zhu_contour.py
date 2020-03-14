import numpy as np
import cv2

def u_to_cnt(u):
    """
    Attention!
    Result is int32 cnt!
    """
    return np.array([[np.real(u),
                      np.imag(u)
                     ]]).T.reshape((-1,1,2)).astype(np.int32)

def cnt_to_u(contour):
        cnt = contour.reshape((-1,2))
        return cnt[:,0] + 1j * cnt[:,1]

def binarize(image):
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
    img = binarize(image)
    contours, _ = cv2.findContours(img,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not get_all:
        measure = [cv2.contourArea(cnt) for cnt in contours]
        contours = [contours[np.argmax(measure)]]
    return [cnt_to_u(cnt) for cnt in contours]

def add_middles(u, iters=1): ### можно переписать эффективнее для iters > 1
    u0 = u.copy()
    for iter_num in range(iters):
        u1 = []
        for i in range(len(u0)):
            u1 += [u0[i], (u0[i]+u0[(i+1)%len(u0)])/2]
        u0 = u1.copy()
    return np.array(u1,dtype=complex)
    
def index_neighbors(u, z, delta = 5):
    if delta is None:
        return np.arange(len(u))
    u_step = np.array(list(u[1:])+[u[0]])
    tmp = max(np.min(np.abs(u_step-u)),1)
    delta_new = delta * tmp ### было /
    return np.arange(len(u))[np.abs(u-z) <= delta_new]