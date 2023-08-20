import os
import socket
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from pcfv import UNet, train_loop, set_normalization, plot_images, plot_images
from torch.utils.data import DataLoader
from datasets import AxialNPY, CoronalNPY, SaggitalNPY
from skimage.metrics import peak_signal_noise_ratio as psnr
from test_loops import test_loop_coronal_npy, test_loop_saggital_npy
from tifffile import imsave
from msd_pytorch import MSDRegressionModel

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

saving_dir = 'L291_result'
print(f"working dir is {saving_dir}")

if not Path(saving_dir).exists():
    print("Creating folder for saving directory")
    Path(saving_dir).mkdir()

# define the model
model = MSDRegressionModel(1, 1, 100, 1, dilations=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
train_batch_size = 1
workers = 16
epochs = 200

# define the training and test dataset
training_data = AxialNPY('recon_low_vertical.npy',
                         'recon_low_horizontal.npy',
                         'recon_low.npy', augmentation=False)
train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True,num_workers=workers)

test_data1 = CoronalNPY('recon_low_test.npy')
test_data2 = SaggitalNPY('recon_low_test.npy')

test_dataloader1 = DataLoader(test_data1, batch_size=1, shuffle=False,num_workers=workers//2)
test_dataloader2 = DataLoader(test_data2, batch_size=1, shuffle=False,num_workers=workers//2)

# set the normalization, important for msd network
# refer to Pelt, D.M., Sethian, J.A.: A mixed-scale dense convolutional neural network for image analysis.
# Proceedings of the National Academy of Sciences 115(2), 254–259 (2018)
set_normalization(model, train_dataloader)
model = model.net

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
mse_loss = torch.nn.MSELoss()

model = model.to(device)

training_losses = []
validate_losses = []

# intermidate results
intermediate_folder = os.path.join(saving_dir,"sr_intermidate")
if not Path(intermediate_folder).exists():
    print("Creating folder for intermediate results")
    Path(intermediate_folder).mkdir()

# image to trace the training process
inter_x, inter_y = training_data[10]
inter_x_cuda = torch.from_numpy(np.expand_dims(inter_x, (0))).float().to(device)
vmin, vmax = inter_y.min(), inter_y.max()

# test image
inter_x_test = test_data1[len(test_data1)//2]
imsave(os.path.join(saving_dir,'test_org.tif'), inter_x_test.astype(np.float32)[0])
inter_x_test_cuda = torch.from_numpy(np.expand_dims(inter_x_test, (0))).float().to(device)

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    loss2 = train_loop(train_dataloader, model, optimizer, mse_loss, device)
    training_losses.append(loss2)
    if t==0 or (t+1) % 5 == 0:
        inter_prey_cuda = model(inter_x_cuda)
        inter_prey = inter_prey_cuda.detach().cpu().numpy()[0]
        fig = plot_images(inter_x[0], inter_prey[0], inter_y[0],
                          style=plt.gray(), t1="original", t2="intermediate result",
                          t3="ground truth", subposition=(1, 3), vmin=vmin, vmax=vmax, range=vmax - vmin,
                          show_image=False,
                          x1=f"psnr:{psnr(inter_x[0], inter_y[0], data_range=vmax - vmin):>.2f}dB",
                          x2=f"{psnr(inter_prey, inter_y, data_range=vmax - vmin):>.2f}dB",
                          width = 455.24408 / 72, height = 455.24408 / 72 /3
                          )
        fig.savefig(os.path.join(intermediate_folder, f"intermediate_epoch_{t + 1}.png"))

        inter_prey_test_cuda = model(inter_x_test_cuda)
        inter_prey_test = inter_prey_test_cuda.detach().cpu().numpy()[0]
        imsave(os.path.join(intermediate_folder, f"test_epoch_{t + 1}.tif"),inter_prey_test)

    torch.save(model.state_dict(), os.path.join(saving_dir,"sr.pt"))

# load trained model
# model.load_state_dict(torch.load(os.path.join(saving_dir,"sr.pt")))

fig = plt.figure(frameon=True)
plt.plot(training_losses, '-')
plt.xlabel('epoch')
plt.ylabel('mse loss')
plt.legend(['Train'])
plt.title('Train loss')
fig.savefig(os.path.join(saving_dir, "train_loss.png"))

test_loop_coronal_npy(test_dataloader1, model, mse_loss, os.path.join(saving_dir,'output_cor.npy'), device, False)
test_loop_saggital_npy(test_dataloader2, model, mse_loss, os.path.join(saving_dir,'output_sag.npy'), device, False)