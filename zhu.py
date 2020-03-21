import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab
import os
import sys

import zhu_contour
import zhu_symmetry
import zhu_draw

def save_results(path, expert_mode = False, **kwargs):  
    """
    input: 
    path --- path to folder with images (any format: png, bmp, jpg, ...). 
    expert_mode --- True => after every image you should tape your opinion: 
            If it contains 'p' or 'P' letter, image will be marked as plane.
                        
    output:   
    None
    """
    names = os.listdir(path='./'+path)
    cols = [
            'area',
            'white_area',
            'Q',
            'angle',
            'symmetry',
            'big_object'
           ]
    if expert_mode:
        cols += ['plane']
    df = pd.DataFrame([],index = names, columns = cols)
        
    plot_f_abs = False
    margin = 10
    Q_thresh = np.inf
    true_Q_thresh = 1
    alpha = 0.01
    beta = 0.99
    delta = 2.5
    area_thresh = 0.12
    theta_thresh = 10
    
    prepare_scene()
    for name in names:
        img = binary(cv2.imread(folder+'/'+name,0))
        imshow_bw(img,name,'gray')
        plt.show()
        u0, tmp_img = external_cnt(img)
        if plot_f_abs:
            u = add_middles(u0)
            vec = - np.min(np.real(u)) - 1j*np.min(np.imag(u)) + (1+1j)*margin
            u += vec
            f = np.fft.fft(u)
            values = np.log(np.abs(f)) / np.log(2)
            min_val = -15
            values[values < min_val] = min_val
            plt.title('Fourier coefficients importance')
            plt.xlabel('l (Fourier coefficient number)')
            ticks = np.arange(np.min(values), np.max(values)+5, 5).astype(float)
            plt.yticks(ticks, np.round(2 ** ticks, 3))
            plt.grid()
            print(len(u))
            plt.plot(values,label = '|f|')
            plt.legend()
            if output_image_format is not None:
                plt.savefig(name+'_abs.' + output_image_format, 
                            format = output_image_format)
            plt.show()
        if expert_mode:
            imshow_bw(img)
            plt.show()
            imshow_bw(tmp_img)
            plt.show()
        area = cv2.contourArea(u_to_cnt(u0))/len(img)/len(img[0])
        white_area = cv2.contourArea(u_to_cnt(u0))/np.sum(img == 255)
        tmp_img1, x,y,theta,thr1,ind1 = another_sym_line(u0, 
                                                         show=False, 
                                                         plot_thr=True, 
                                                         name=res_folder+'/'+name, #save graphics there
                                                         delta=delta, 
                                                         line=3,
                                                         thresh=Q_thresh, 
                                                         alpha=alpha, 
                                                         beta=beta, 
                                                         draw_points=False) #by hull points
        angle = (theta - np.pi/2) / np.pi * 180
        title = 'Q = ' + str(round(thr1, 3))
        cmap = 'gray'
        if thr1 > true_Q_thresh:
            cmap = 'magma'
#         elif abs(angle) > theta_thresh:
#             cmap = 'Set1'
        imshow_bw(tmp_img1,title,cmap)
        if output_image_format is not None:
            plt.savefig(res_folder+'/'+name[:-4]+'_result.'+output_image_format,format=output_image_format)
        vec = [area,white_area,thr1,
               angle,thr1<=true_Q_thresh,area>=area_thresh]
        plt.show()
        if expert_mode:
            res = input()
            is_plane = 'p' in res or 'P' in res
            vec += [is_plane]
        df.loc[name] = vec
    table_path = path + '/masks.csv'
    open(table_path, 'a').close()
    df.to_csv(table_path,columns=df.columns)

def many_objects(Q, folder, res_folder, output_image_format = None):  

    """
    input: 
    folder --- path to folder with images (any format: png, bmp, jpg, ...). Folder should contain ONLY images.
    res_folder --- path to folder for saving results.
    expert_mode --- if if is True, after every image you should tape your opinion about it. 
                        If it contains 'p' or 'P' letter, image will be marked as plane.
    output_table_format --- it should be only csv or xls.
    output_image_format --- if it is not None, image in chosen format will be saved 
                            (eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff).                            
    output:   None
    """
    prepare_scene()
       
    imgs = []
    for name in names:
        res = cv2.imread(folder+'/'+name,0)
        img = binary(res)
#         imshow_bw(img,name,'gray')
#         plt.show()
        for i, (u0, tmp_img) in enumerate(external_cnt(img, many = True)):
            area = cv2.contourArea(u_to_cnt(u0))/len(img)/len(img[0])
            if area < area_thresh:
                continue
            white_area = cv2.contourArea(u_to_cnt(u0))/np.sum(img == 255)
            tmp_img1, x, y, theta, thr1, ind1 = another_sym_line(u0, 
                                                                 show=False, 
                                                                 plot_thr=False, 
                                                                 #name=name, 
                                                                 delta=delta, 
                                                                 line=3,
                                                                 thresh=Q_thresh, 
                                                                 alpha=alpha, 
                                                                 beta=beta, 
                                                                 draw_points=False)
            angle = (theta - np.pi/2) / np.pi * 180
            title = 'Q = ' + str(round(thr1, 3))
            cmap = 'gray'
#             if thr1 > true_Q_thresh:
#                 cmap = 'magma'
            imgs.append((tmp_img1.copy(), 
                         title, 
                         cmap, 
                         thr1))
#             imshow_bw(tmp_img1, title, cmap)
#             if output_image_format is not None:
#                 plt.savefig(res_folder + '/' + 
#                             name[:-4] + '_result' + str(i)
#                             + '.' + output_image_format,
#                             format=output_image_format)
#             vec = {'area': area,
#                    'white_area': white_area,
#                    'thr': thr1,
#                    'angle': angle,
#                    'sym?': thr1 <= true_Q_thresh,
#                    'big?': area >= area_thresh
#                   }
#             print(vec)
#             plt.show()        
    imgs = sorted(imgs, key = lambda x: x[-1])
    print(len(imgs))
    cols = 5
    rows = (len(imgs) + cols - 1) // cols
    f, axarr = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    plt.setp(axarr, xticks=[], yticks=[])
    i = 0
    for row in range(rows):
        for col in range(cols):
            if i >= len(imgs):
                break
            img, title, cmap, thr = imgs[i] 
            imshow_bw(img, title, cmap, axarr[row][col])
            i += 1
    if output_image_format is not None:
        plt.savefig(res_folder + '/' + 
                    'result.' + output_image_format,
                    format = output_image_format)
    plt.show()
