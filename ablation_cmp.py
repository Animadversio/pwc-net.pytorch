from PWC_src.pwc_ablation import PWC_Net as PWC_Net_abl
from PWC_src.pwc_ablation_ups import PWC_Net as PWC_Net_abl_ups
from PWC_src import PWC_Net
import torch
import cv2
import numpy as np
import matplotlib.pylab as plt
from visualization import visualize_pyr, visualize_samples
from datasets import SintelDataset, resize_pyramid
pwc = PWC_Net_abl(model_path='models/sintel.pytorch')
pwc_ups = PWC_Net_abl_ups(model_path='models/sintel.pytorch').cuda()
pwc_sd = PWC_Net(model_path='models/sintel.pytorch').cuda()
# pwc = PWC_Net(model_path='models/train_demo.pytorch')
pwc = pwc.cuda()
#%%
SintelClean = SintelDataset(render="clean", torchify=True, cropsize=None)
im1, im2, trflow = SintelClean[1000]
# with torch.no_grad():
#     predflow_pyr = pwc(im1.unsqueeze(0).cuda(), im2.unsqueeze(0).cuda())
# trflow_pyr = resize_pyramid(trflow)
# figh = visualize_pyr(predflow_pyr, trflow_pyr, im1=im1, im2=im2)[0]
# figh.show()
with torch.no_grad():
    predflow_pyr = pwc_sd(im1.unsqueeze(0).cuda(), im2.unsqueeze(0).cuda())
trflow_pyr = resize_pyramid(trflow)
figh = visualize_pyr(predflow_pyr, trflow_pyr, im1=im1, im2=im2)[0]
figh.show()
#
with torch.no_grad():
    predflow_pyr = pwc_ups(im1.unsqueeze(0).cuda(), im2.unsqueeze(0).cuda())
trflow_pyr = resize_pyramid(trflow)
figh = visualize_pyr(predflow_pyr, trflow_pyr, im1=im1, im2=im2)[0]
figh.show()