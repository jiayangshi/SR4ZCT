import numpy as np
from scipy import interpolate

def resample_img_ax0(img, org_res=0.7421875, org_overlap=0, exp_res=3, exp_overlap=2, keep_dim=True, num_points_overlap=9):
    (w, h) = img.shape
    length = org_res + (w -1) * (org_res - org_overlap)
    num_new = 1 + int((length-exp_res) / (exp_res - exp_overlap))
    x_old = np.linspace(org_res/2, length-org_res/2, w, endpoint=True)
    x_new = np.linspace(exp_res/2, length-exp_res/2, num_new, endpoint=True)
    if keep_dim:
        image_new = np.zeros((w,h))
    else:
        image_new = np.zeros((len(x_new), h))
    vmin, vmax = img.min(), img.max()
    # go for every column
    for i in range(h):
        f = interpolate.interp1d(x_old, img[:,i], kind='linear',fill_value='extrapolate')
        y_new = f(x_new)
        if exp_overlap > 0:
            for j in range(num_points_overlap // 2):
                # average the pixel values
                x_new_l = x_new - exp_res / num_points_overlap * (j + 1)
                x_new_r = x_new + exp_res / num_points_overlap * (j + 1)
                y_new_l = f(x_new_l)
                y_new_r = f(x_new_r)
                y_new += (y_new_l + y_new_r)
            y_new /=num_points_overlap
        if keep_dim:
            f2 = interpolate.interp1d(x_new, y_new, kind='linear',fill_value='extrapolate')
            y_new = f2(x_old)
        image_new[:,i] = y_new
    image_new[image_new<vmin] = vmin
    image_new[image_new>vmax] = vmax
    return image_new

def resample_img_ax1(img,org_res=0.7421875,org_overlap=0,exp_res=3,exp_overlap=2,keep_dim=True, num_points_overlap=9):
    (w, h) = img.shape
    length = org_res + (h -1) * (org_res - org_overlap)
    num_new = 1 + int((length-exp_res) / (exp_res - exp_overlap))
    x_old = np.linspace(org_res/2, length-org_res/2, h, endpoint=True)
    x_new = np.linspace(exp_res/2, length-exp_res/2, num_new, endpoint=True)
    if keep_dim:
        image_new = np.zeros((w,h))
    else:
        image_new = np.zeros((w,len(x_new)))
    vmin, vmax = img.min(), img.max()
    # go for every row
    for i in range(w):
        x_old = np.arange(org_res/2,length-org_res/4,org_res-org_overlap)
        x_new = np.arange(exp_res/2,length-exp_res/4,exp_res-exp_overlap)
        f = interpolate.interp1d(x_old, img[i], kind='linear',fill_value='extrapolate')
        y_new = f(x_new)
        if exp_overlap > 0:
            for j in range(num_points_overlap // 2):
                # average the pixel values
                x_new_l = x_new - exp_res / num_points_overlap * (j + 1)
                x_new_r = x_new + exp_res / num_points_overlap * (j + 1)
                y_new_l = f(x_new_l)
                y_new_r = f(x_new_r)
                y_new += (y_new_l + y_new_r)
            y_new /= num_points_overlap
        if keep_dim:
            f2 = interpolate.interp1d(x_new, y_new, kind='linear',fill_value='extrapolate')
            y_new = f2(x_old)
        image_new[i] = y_new
    image_new[image_new<vmin] = vmin
    image_new[image_new>vmax] = vmax
    return image_new