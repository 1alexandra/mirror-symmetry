import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import os
import sys


def save_results(folder, res_folder, expert_mode = False, output_table_format = 'csv', output_image_format = None):  

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
    
    if output_table_format not in ['xls','csv']:
        print('Output table format should be xls or csv.')
    
    names = os.listdir(path='./'+folder)
    cols = ['area','white_area','Q',
            'angle','symmetry','big_object']
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
    if not os.path.isdir(res_folder):
        os.mkdir(res_folder)
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
    table_path = res_folder+'/masks.'+output_table_format
    open(table_path, 'a').close()
    if output_table_format == 'xls':
        df.to_excel(table_path,columns=df.columns)
    elif output_table_format == 'csv':
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
    
    names = os.listdir(path='./'+folder)
    cols = ['area','white_area','Q',
            'angle','symmetry','big_object']
    df = pd.DataFrame([],index = names, columns = cols)
        
    Q_thresh = np.inf
    true_Q_thresh = Q
    alpha = 0
    beta = 1
    delta = 2.5
    area_thresh = 0.01
    theta_thresh = 10
    if not os.path.isdir(res_folder):
        os.mkdir(res_folder)
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
    
def binary(img):
    img = cv2.GaussianBlur(img, (5,5), 0)
    img_new = np.array(img,dtype=np.uint8)
    if img[0,0] == 255:
        img_new = 255-img_new
    img_new *= (int)(255 / np.max(img_new))
    if len(np.unique(img)) != 2:
        img_new = cv2.adaptiveThreshold(img_new,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,0)
    return img_new


def external_cnt(img, line = 1, many = False):
    img1 = binary(img)
    contours, hierarchy = cv2.findContours(img1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not many:
        mesure = [cv2.contourArea(cnt) for cnt in contours]
        cnt = contours[np.argmax(mesure)].reshape((-1,2))
        u = cnt[:,0] + 1j * cnt[:,1]
        tmp_img = np.zeros(img.shape)
        cv2.drawContours(tmp_img,contours,np.argmax(mesure),255,line)
        return u, tmp_img
    ans = []
    for i, cur_cnt in enumerate(contours):
        cnt = cur_cnt.reshape((-1,2))
        u = cnt[:,0] + 1j * cnt[:,1]
        tmp_img = np.zeros(img.shape)
        cv2.drawContours(tmp_img,contours,i,255,line)
        ans.append((u, tmp_img))
    return ans


def u_to_cnt(u):
    return np.array([[np.real(u),np.imag(u)]]).T.reshape((-1,1,2)).astype(np.int32)


def add_middles(u, iters=1):
    u0 = u.copy()
    for iter_num in range(iters):
        u1 = []
        for i in range(len(u0)):
            u1.append(u0[i])
            if i!=len(u0)-1:
                u1.append((u0[i]+u0[i+1])/2)
            else:
                u1.append((u0[-1]+u0[0])/2)
        u0 = u1.copy()
    return np.array(u1,dtype=complex)


def u_delta(u,x,y,delta):
    if delta is None:
        return np.arange(len(u))
    u_step = np.array(list(u[1:])+[u[0]])
    tmp = max(np.min(np.abs(u_step-u)),0.5)
    delta_new = delta/tmp
    z = x + 1j * y
    return np.arange(len(u))[np.abs(u-z) <= delta_new]


def new_start_point(f, s, f_ind=None):
    N = len(f)
    if f_ind is None:
        f_ind = np.arange(1,N)
    pows = -1j*2*np.pi/N*f_ind*s
    return f[f_ind]*np.exp(pows)

def filter_f(g, alpha = 0, beta = 1, ret_zero = False):
    a1 = np.min(np.abs(g[1:]))
    a2 = np.max(np.abs(g[1:]))
    eps = a1+(a2-a1)*alpha
    N = len(g)
    garms = N*beta   
    tmp = np.arange(N)
    index = (np.abs(g)>=eps)*(np.logical_or(tmp<=garms/2,tmp>=N-garms/2))
    if not ret_zero:
        index *= index > 0
    return tmp[index]

def find_theta(dots_arr):
    a = np.real(dots_arr)
    b = np.imag(dots_arr)
    k1 = np.sum(b*b)
    k2 = np.sum(a*a)
    k3 = 2*np.sum(a*b)
    f = lambda t: k1*np.cos(t)**2+k2*np.sin(t)**2+2*k3*np.sin(t)*np.cos(t)
    if k1 == k2:
        if k3 == 0:
            print('find_theta error: no way')
            return 0
        t1 = np.pi/4
    elif k3 == 0:
        t1 = 0
    else:
        t1 = 0.5 * np.arctan(k3/(k1-k2))
    t2 = t1 + 0.5 * np.pi
    if f(t1)<f(t2):
        if f(0)>f(t1):
            return -t1 % np.pi
    else:
        if f(0)>f(t2):
            return -t2 % np.pi
    return 0
    
def rotate_calc(dots_arr, N, ret_theta = False, ret_abs_sum = False, ret_sum_abs = False):
    theta = find_theta(dots_arr)
    dots_new = dots_arr * np.exp(-1j*theta)
    a = np.real(dots_new)
    b = np.imag(dots_new)
#     thr =  (np.sum(b*b)/N)**0.5
    thr = (np.sum(b*b))**0.5 / N
    if ret_theta:
        return thr, theta
    if ret_abs_sum:
        return np.abs(np.sum(b)/N)
    if ret_sum_abs:
        return np.sum(np.abs(b)/N)
    return thr


def hull_based_index(u,delta):
    if delta is None:
        return np.arange(len(u)),np.array([],dtype=int)
    cnt = np.array([np.real(u) * 2, np.imag(u) * 2], dtype=int).T
    cnt = cnt.reshape((-1,2))
    hull0 = cv2.convexHull(cnt, returnPoints=False)
    centroid = np.mean(u)
    hull0 = np.ravel(hull0)
    hull_u = u[hull0]
    middles0 = []
    for i in range(len(hull0)):
        next_i = (i+1)%len(hull_u)
        bet_z = np.mean([hull_u[i],hull_u[next_i]])
        re_z1, im_z1 = np.real(bet_z), np.imag(bet_z)
        re_z2, im_z2 = np.real(centroid), np.imag(centroid)
        end = hull0[i]
        start = hull0[next_i]
        if start > end:
            ind = np.concatenate((np.arange(start,len(u)),np.arange(0,end)))
        else:
            ind = np.arange(start,end)
        tmp = u[ind]
        dists_min = np.argmin(np.abs((im_z2-im_z1)*np.real(tmp)-(re_z2-re_z1)*np.imag(tmp)+im_z1*re_z2-im_z2*re_z1))
        ind = ind[dists_min]
        middles0.append(ind)
    middles0 = np.array(middles0)
    hull1 = np.unique(np.ravel(np.array([hull0,middles0],dtype = int).T))
    hull_delta = []
    for ind in hull1:
        el = u[ind]
        hull_delta += list(u_delta(u,np.real(el),np.imag(el),delta))
    hull2 = np.unique(np.ravel(np.array(hull_delta,dtype = int).T))
    return len(u)-1-np.arange(len(u))[hull2], len(u)-1-np.arange(len(u))[hull1]


def another_sym_line(u1, show=True, plot_thr=False, delta=None, line=1, thresh=100,
                     name=None, gauss=False, margin=10,alpha=0,beta=1,draw_points=True):
    u = add_middles(u1)
    vec = - np.min(np.real(u)) - 1j*np.min(np.imag(u)) + (1+1j)*margin
    u += vec
    f = np.fft.fft(u)
    N = len(u)
    tmp_img = draw_ifft(f,False,line)
    by_hull, hull = hull_based_index(u,delta)
    f_ind = filter_f(f,alpha,beta)
    thrs = [rotate_calc(new_start_point(f,s,f_ind),N) for s in by_hull]
    first_true_index = by_hull[np.argmin(thrs)]
    x, y = np.real(u[first_true_index]), np.imag(u[first_true_index])
    ind = u_delta(u,x,y,delta)
    thrs = [rotate_calc(new_start_point(f, s), N) for s in ind]
    true_index = ind[np.argmin(thrs)]
    thr, theta = rotate_calc(new_start_point(f,true_index),N,True)
    x = -1
    y = -1
    if thr<thresh:
        x, y = np.real(u[-true_index]), np.imag(u[-true_index])
        v_x, v_y = np.cos(theta), np.sin(theta)
        c = (int)(800/tmp_img.shape[0])
        tmp_img = draw_ifft(f*c,False,line)   
        w, h = tmp_img.shape
        if draw_points:
            tmp = np.zeros((w,h,3),dtype=int)*255
            for inddd in range(3):
                tmp[:,:,inddd] = tmp_img.copy()
            tmp_img = tmp
        k = max(tmp_img.shape)
        cv2.line(tmp_img,(c*int(x),c*int(y)),(c*int(x-k*v_x),c*int(y-k*v_y)),(255,255,255),line)
        cv2.line(tmp_img,(c*int(x),c*int(y)),(c*int(x+k*v_x),c*int(y+k*v_y)),(255,255,255),line)
        if draw_points:
            for el in by_hull:
                x, y = np.real(u[-el]), np.imag(u[-el])
                cv2.circle(tmp_img,(c*int(x),c*int(y)),line,(0,255,255),line)
            for el in hull:
                x, y = np.real(u[-el]), np.imag(u[-el])
                cv2.circle(tmp_img,(c*int(x),c*int(y)),line*2,(255,255,0),line)
    if plot_thr:
        plt.xlabel('p (point index)')
        plt.ylabel('asymmetry measure')
        plt.grid()
        plt.title('min(Q) = '+str(round(thr,3)))
        thrs = [rotate_calc(new_start_point(f, s), N) for s in range(N)]
        plt.plot(thrs,label = 'Q(p)')
        plt.plot([true_index],[thr],'ro', label = 'min Q(p)')
        
#         thrs_abs_sum = [rotate_calc(new_start_point(f, s), N, ret_abs_sum = True) for s in range(N)]
#         plt.plot(thrs_abs_sum,label = 'abs(sum)')
#         ind_min = np.argmin(thrs_abs_sum)
#         plt.plot([ind_min],[thrs_abs_sum[ind_min]],'bo', label = 'min abs(sum)')
#         thrs_sum_abs = [rotate_calc(new_start_point(f, s), N, ret_sum_abs = True) for s in range(N)]
#         plt.plot(thrs_sum_abs,label = 'sum(abs)')
#         ind_min = np.argmin(thrs_sum_abs)
#         plt.plot([ind_min],[thrs_sum_abs[ind_min]],'bo', label = 'min sum(abs)')
        
        plt.legend()
        if not name is None:
            plt.savefig(name[:-4]+'_q.png',format='png',bbox_inches='tight')
        plt.show()
#         plt.xlabel('coefficients')
#         plt.ylabel('importance')
#         plt.title('Fourier descriptor importance')
#         plt.plot(np.log(np.abs(f))[1:], label='log|f|')
#         plt.legend()
#         plt.show()
    if show:
        if i>0:
            draw_complex(new_start_point(f,true_index),True,False,'F, l>0',False)
            plt.savefig('plane_f'+str(i)+'.png',format='png')
            plt.show()
            if gauss:
                tmp_img = cv2.GaussianBlur(tmp_img, (3,3), 0)
            plt.imsave('plane_line'+str(i)+'.png',255-tmp_img,cmap='gray',format = 'png')
        plt.imshow(255-tmp_img,cmap='gray')
        plt.show()
    return tmp_img, x-np.real(vec), y-np.imag(vec), theta, thr, -true_index


def draw_ifft(f,show = False,line=1,margin=10):
    u = np.fft.ifft(f)
    h = (int)(np.max(np.real(u))) + margin
    w = (int)(np.max(np.imag(u))) + margin
    tmp_img = np.zeros((w,h))
    cnt = u_to_cnt(u)
    cv2.drawContours(tmp_img,[cnt],0,255,line)
    if show:
        plt.imshow(tmp_img,cmap='gray')
        plt.show()
    return tmp_img


def draw_complex(u, draw_zero = False, draw_line = False, label = '',show = True):
    x1 = min(np.min(np.real(u)),np.min(np.imag(u)))-50
    x2 = max(np.max(np.real(u)),np.max(np.imag(u)))+50
    plt.xlim(x1,x2)
    plt.ylim(x1,x2)
    plt.axis('equal')
    if draw_line:
        x = np.zeros(len(u)+1)
        x[:-1] = np.real(u)
        x[-1] = np.real(u[0])
        y = np.zeros(len(u)+1)
        y[:-1] = np.imag(u)
        y[-1] = np.imag(u[0])
        plt.plot(x,y,color='gray')
    plt.plot(np.real(u),np.imag(u),'go',label = label)
    if draw_zero:
        plt.plot([0],[0],"ro",label = 'zero')
    if label!='':
        plt.legend()
    if show:
        plt.show()

        
def imshow_bw(img, title = '', cmap = 'gray', ax = None):
    if ax is None:
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(255-img, cmap=cmap)
    else:
        ax.set_title(title,fontsize= 30)
        ax.imshow(255-img, cmap=cmap)


if __name__ == "__main__":
    expert_mode = False
    output_table_format = 'csv'
    output_image_format = None
    folder = sys.argv[1]
    res_folder = sys.argv[2]
    if len(sys.argv) > 3:
        expert_mode = sys.argv[3] in ['True','true','1','on']
    if len(sys.argv) > 4:
        output_table_format = sys.argv[4]
    if len(sys.argv) > 5:
        output_image_format = sys.argv[5]
    if len(sys.argv) > 6:
        print('Other args ignored.')
    save_results(folder,res_folder,expert_mode,output_table_format,output_image_format)
    sys.exit()