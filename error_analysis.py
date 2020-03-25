import pickle
import numpy as np
import matplotlib.pylab as plt
#%%
#err_data = pickle.load(open("..\\Sintel_clean_model_ups_abl_ep117_EPE_lvl.pk", "rb"))
err_data = pickle.load(open("..\\Sintel_clean_model_sintel_EPE_lvl.pk", "rb"))
mepe_hrs = err_data['mepe_hrs']
mepe_level = err_data['mepe_level']
for l in range(1,6):
    print(np.corrcoef(mepe_hrs, mepe_level[l])[0,1])
#%%
from datasets import SintelDataset, warp_image, resize_pyramid
SintelImage = SintelDataset(render="clean", torchify=False)
im1, im2, flow = SintelImage[10]
#%%
im1_warp = warp_image(im1 / 255.0, 2*flow)
#%
plt.figure(figsize=[8,4])
plt.subplot(221)
plt.imshow(im1);plt.axis("off")
plt.subplot(222)
plt.imshow(im2)
plt.subplot(223)
plt.imshow(im1_warp)
plt.show()
#%%
#%%
grid_ref = np.zeros(im1.shape[:2])[:,:,np.newaxis]
grid_ref[::10, :, :] = 1.0
grid_ref[:, ::10, 0] = 1.0
grid_warp = warp_image(grid_ref, 0.5*flow)
#%%
plt.figure(figsize=[10,3])
plt.subplot(221)
plt.imshow(grid_ref[:,:,0], cmap="gray");plt.axis("off")
plt.subplot(222)
plt.imshow(grid_warp[:,:,0], cmap="gray");plt.axis("off")
plt.show()
#%%
plt.figure(figsize=[10,3])
plt.imshow(grid_warp[:,:,0], cmap="gray");plt.axis("off")
import cv2
cv2.imwrite("..\\Results\\warpped_grid.jpg", grid_warp[:,:,0])
plt.show()
#%%
cv2.GaussianBlur()