import os
from pydicom import dcmread
import numpy as np
from tqdm import tqdm
from utils import resample_img_ax0, resample_img_ax1


# Path to the folder containing the DICOM files
folder = 'L291/full_3mm_sharp/'

files = os.listdir(folder)
files.sort()

# only cut the lung area, for L291 3mm slices, the lung area is from 0 to 140
slices = 140 
recon_low = np.zeros((slices, 512, 512), dtype=np.float32)

print("Reading DICOM files...")
for i in tqdm(range(slices)):
    recon_low[i] = dcmread(os.path.join(folder, files[i])).pixel_array

# normalize the volume
vmin, vmax = recon_low.min(), recon_low.max()
recon_low = (recon_low - vmin)/ (vmax - vmin)

np.save('recon_low.npy', recon_low)

# down and re-upscale to create training data
# the actual vertical resolution is 3mm with 1 mm overlap
# we create the virtual 3mm/1mm training images from the original 0.7421875mm/0mm axial images
# the axial image is first downscaled to 3mm/1mm, 
# and upscaled to 0.7421875mm/0mm to have the same size as the original image with blurry preserved
recon_low0 = np.zeros((slices, 512, 512), dtype=np.float32)
recon_low1 = np.zeros((slices, 512, 512), dtype=np.float32)

for i in tqdm(range(slices)):
    img1 = resample_img_ax0(recon_low[i], org_res=0.7421875, org_overlap=0, exp_res=3, exp_overlap=1, keep_dim=True)
    img2 = resample_img_ax1(recon_low[i], org_res=0.7421875, org_overlap=0, exp_res=3, exp_overlap=1, keep_dim=True)
    recon_low0[i] = img1
    recon_low1[i] = img2

np.save('recon_low_vertical.npy', recon_low0)
np.save('recon_low_horizontal.npy', recon_low1)

# prepare the test volume
# upscale in vertical direction from 3mm/1mm to 0.7421875mm/0mm
test = resample_img_ax0(recon_low[:,0], org_res=3.0, org_overlap=1.0, exp_res=0.7421875, exp_overlap=0,
                                     keep_dim=False)
recon_low_test = np.zeros((test.shape[0],512,512),dtype=np.float32)
print(f'recon_low_test shape {recon_low_test.shape}')

for i in tqdm(range(512)):
    img1 = recon_low[:,i,:]
    recon_low_test[:,i,:] = resample_img_ax0(img1, org_res=3.0, org_overlap=1.0, exp_res=0.7421875, exp_overlap=0,
                                     keep_dim=False)

np.save('recon_low_test.npy', recon_low_test)