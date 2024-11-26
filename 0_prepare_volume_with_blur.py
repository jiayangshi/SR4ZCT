import numpy as np
from tqdm import tqdm
from utils import resample_img_ax0,resample_img_ax1, resample_img_ax0_with_blurr, resample_img_ax1_with_blurr
from tifffile import imread
from pathlib import Path
import os
import skimage.metrics as skm

imgs = imread('mouse_embryo.tif')
recon_low = img.copy()

hor_res = 1
ver_res = 4
sigma_blurr = 1.5

save_dir = 'mouse_embryo'
if not Path(save_dir).exists():
    print("Creating folder for saving directory")
    Path(save_dir).mkdir()

# normalize the volume, somtimes normalization helps with convergence of training
# but not always necessary
# use the vmin, vmax to scale back to original if needed
vmin, vmax = recon_low.min(), recon_low.max()
recon_low = (recon_low - vmin)/ (vmax - vmin)

np.save(os.path.join(save_dir, 'recon_low.npy'), recon_low)

recon_low0 = np.zeros(imgs.shape, dtype=np.float32)
recon_low1 = np.zeros(imgs.shape, dtype=np.float32)

for i in tqdm(range(imgs.shape[0])):
    img1 = resample_img_ax0_with_blurr(recon_low[i], org_res=hor_res, org_overlap=0, exp_res=ver_res, exp_overlap=0, keep_dim=True, blurr_sigma=sigma_blurr)
    img2 = resample_img_ax1_with_blurr(recon_low[i], org_res=hor_res, org_overlap=0, exp_res=ver_res, exp_overlap=0, keep_dim=True, blurr_sigma=sigma_blurr)
    
    recon_low0[i] = img1
    recon_low1[i] = img2

np.save(os.path.join(save_dir, 'recon_low_vertical.npy'), recon_low0)
np.save(os.path.join(save_dir,'recon_low_horizontal.npy'), recon_low1)

test = resample_img_ax0(recon_low[:,0], org_res=ver_res, org_overlap=0, exp_res=hor_res, exp_overlap=0,
                                     keep_dim=False)
recon_low_test = np.zeros((test.shape[0],imgs.shape[1],imgs.shape[2]),dtype=np.float32)
print(f'recon_low_test shape {recon_low_test.shape}')

for i in tqdm(range(test.shape[1])):
    recon_low_test[:,i,:] = resample_img_ax0(recon_low[:,i,:], org_res=ver_res, org_overlap=0, exp_res=hor_res, exp_overlap=0,
                                     keep_dim=False)

np.save(os.path.join(save_dir, 'recon_low_test.npy'), recon_low_test)