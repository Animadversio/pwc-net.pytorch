
import cv2
import numpy as np
import matplotlib.pylab as plt

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch import optim
from PWC_src import PWC_Net
from PWC_src import flow_to_image, read_flow, write_flow, flow_error, segment_flow

from os import listdir, makedirs
from os.path import join
from imageio import imwrite
from scipy.ndimage import zoom
import time
#%%# Build model
FLOW_SCALE = 20.0
# pwc = PWC_Net(model_path='models/sintel.pytorch')
pwc = PWC_Net(model_path='models/chairs-things.pytorch')
# pwc = PWC_Net(model_path='models/train_demo.pytorch')
pwc = pwc.cuda()
pwc.eval()
# pwc.train()
#%%
from datasets import SintelDataset, FlyingChairDataset, resize_pyramid, DataLoader
SintelClean = SintelDataset(render="clean", torchify=True, cropsize=(384, 768))
#%%
loader = DataLoader(dataset=SintelClean, batch_size=1, shuffle=False, drop_last=True)
for sample in loader:
    im1, im2, flow = sample
    flow_infer = FLOW_SCALE * pwc(im1.cuda(), im2.cuda())
    visualize_batch_samples(im1, im2, flow, flow_infer=flow_infer.cpu())
    break
#%%
pwc.train()
loader = DataLoader(dataset=SintelClean, batch_size=1, shuffle=False, drop_last=True)
for sample in loader:
    im1, im2, flow = sample
    flow_infer_pyr0 = pwc(im1.cuda(), im2.cuda()) # FLOW_SCALE *
    fighs = visualize_pyr(flow_infer_pyr0, resize_pyramid(flow), im1=im1, im2=im2, level=None)
    for figh in fighs:
        figh.show()
    break
#%% Flow_infer_pyr stats
for fl in flow_infer_pyr0:
    for si in range(fl.shape[0]):
        print("samp%d, u: %.2f (%.2f) [%.2f, %.2f]; v: %.2f (%.2f) [%.2f, %.2f]"%
              (si, fl[si, 0, :].mean(), fl[si, 0, :].std(), fl[si, 0, :].min(), fl[si, 0, :].max(),
               fl[si, 1, :].mean(), fl[si, 1, :].std(), fl[si, 1, :].min(), fl[si, 1, :].max(), ))
#%%
#%%
pwc.debug = True
SintelClean = SintelDataset(render="clean", torchify=True, cropsize=None)
#%%
loader = DataLoader(dataset=SintelClean, batch_size=3, shuffle=False, drop_last=True)
for sample in loader:
    im1, im2, flow = sample
    with torch.no_grad():
        flow_infer_pyr0, FeatPyr1, FeatPyr2 = pwc(im1.cuda(), im2.cuda())  # FLOW_SCALE *
    fighs = visualize_pyr(flow_infer_pyr0, resize_pyramid(flow), im1=im1, im2=im2, level=None)
    for figh in fighs:
        figh.show()
    break
#%% Flow_infer_pyr stats
print("Statistics of Inferred Flow")
for si in range(im1.shape[0]):
    for fl in flow_infer_pyr0:
        print("samp%d, u: %.2f (%.2f) [%.2f, %.2f]; v: %.2f (%.2f) [%.2f, %.2f]"%
              (si, fl[si, 0, :].mean(), fl[si, 0, :].std(), fl[si, 0, :].min(), fl[si, 0, :].max(),
               fl[si, 1, :].mean(), fl[si, 1, :].std(), fl[si, 1, :].min(), fl[si, 1, :].max(), ))
    print("\n")
print("Statistics of feature Pyramid")
for si in range(im1.shape[0]):
    for Ft1, Ft2 in zip(FeatPyr1, FeatPyr2):
        print("samp%d, FeatTsr1: %.3f (%.2f) [%.2f, %.2f]; FeatTsr2: %.3f (%.2f) [%.2f, %.2f]"%
              (si, Ft1[si, :, :].mean(), Ft1[si, :, :].std(), Ft1[si, :, :].min(), Ft1[si, :, :].max(),
               Ft2[si, :, :].mean(), Ft2[si, :, :].std(), Ft2[si, :, :].min(), Ft2[si, :, :].max(),))
    print("\n")
#%%
# Seems for batch input to the network the extracted flow at top layer will be exactly the same for different images.

#%% seems the samples are interacting with each other
#%%
SintelClean = SintelDataset(render="clean", torchify=False, cropsize=(384, 768))
#%%
im1, im2, flow = SintelClean[0]
visualize_samples(im1, im2, flow)
plt.show()
#%%
im1 = torch.cat((im1.unsqueeze(0), im1.unsqueeze(0)))
im2 = torch.cat((im2.unsqueeze(0), im2.unsqueeze(0)))
flow = torch.cat((flow.unsqueeze(0), flow.unsqueeze(0)))
with torch.no_grad():
    flow_infer_pyr0, FeatPyr1, FeatPyr2 = pwc(im1.cuda(), im2.cuda())  # FLOW_SCALE *
fighs = visualize_pyr(flow_infer_pyr0, resize_pyramid(flow), im1=im1, im2=im2, level=None)
for figh in fighs:
    figh.show()

print("Statistics of feature Pyramid")
for si in range(im1.shape[0]):
    for Ft1, Ft2 in zip(FeatPyr1, FeatPyr2):
        print("samp%d, FeatTsr1: %.3f (%.2f) [%.2f, %.2f]; FeatTsr2: %.3f (%.2f) [%.2f, %.2f]"%
              (si, Ft1[si, :, :].mean(), Ft1[si, :, :].std(), Ft1[si, :, :].min(), Ft1[si, :, :].max(),
               Ft2[si, :, :].mean(), Ft2[si, :, :].std(), Ft2[si, :, :].min(), Ft2[si, :, :].max(),))
    print("\n")
print("Statistics of Inferred Flow")
for si in range(im1.shape[0]):
    for fl in flow_infer_pyr0:
        print("samp%d, u: %.2f (%.2f) [%.2f, %.2f]; v: %.2f (%.2f) [%.2f, %.2f]"%
              (si, fl[si, 0, :].mean(), fl[si, 0, :].std(), fl[si, 0, :].min(), fl[si, 0, :].max(),
               fl[si, 1, :].mean(), fl[si, 1, :].std(), fl[si, 1, :].min(), fl[si, 1, :].max(), ))
    print("\n")

#%% Pick some random frames and show the result compared to ground truth image
sampi = np.random.randint(len(SintelClean))
print(SintelClean.scene_src[sampi])
im1, im2, trflow = SintelClean[sampi]
trflow_pyr = resize_pyramid(trflow.unsqueeze(0).cuda())
predflow_pyr = pwc(im1.unsqueeze(0).cuda(), im2.unsqueeze(0).cuda())
visualize_pyr(predflow_pyr, trflow_pyr, im1=im1, im2=im2,level=None)
#%%
#%%
tmp_dict = pwc.moduleExtractor.moduleSix.state_dict()
list(tmp_dict.keys())
for param in tmp_dict.keys():
    print("%s: %E"%(param, tmp_dict[param].abs().mean()))
# tmp_dict['0.weight'].abs().mean()

# Note the Six layer feature extractor has no weight! not trained seemingly
# 0.weight: 9.879992E-36
# 0.bias: 2.317401E-03
# 2.weight: 1.095370E-35
# 2.bias: 1.398091E-05
# 4.weight: 4.461260E-33
# 4.bias: 3.139201E-25
# So the six layer's correlation output is very unhelpful.
#%%
from PWC_src.flowlib import *
import matplotlib.pylab as plt
#%% See the reference map
colorwheel = make_color_wheel()
steps = 25
XX, YY = np.meshgrid(range(-steps, steps+1), range(-steps, steps+1))
clr_map = compute_color(XX / steps, YY / steps)
plt.figure(figsize=[5.5, 5])
plt.imshow(clr_map, extent=[-1, 1, -1, 1])
plt.xlabel("Normalized U")
plt.ylabel("Normalized V")
plt.title("Middleburry Flow Color Code")
plt.savefig("..\\Results\\FlowColorCode.png")
plt.show()
#%%

#%%
warp_image

#%%
import time


#%%
im1 = cv2.imread(r"C:\Users\PonceLab\Documents\PWC_net_Binxu\Datasets\Sintel\training\final\alley_1\frame_0002.png")
im2 = cv2.imread(r"C:\Users\PonceLab\Documents\PWC_net_Binxu\Datasets\Sintel\training\final\alley_1\frame_0003.png")
im1 = torch.from_numpy((im1 / 255.).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
im2 = torch.from_numpy((im2 / 255.).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
im1_v = im1.cuda()
im2_v = im2.cuda()
start = time.time()
flow_list = pwc(im1_v, im2_v)
# FLOW_SCALE *
print(time.time() - start)
#%%
for flow in flow_list:
    print(flow.shape, "range(%.3f,%.3f)" % (flow.min(), flow.max()))
# torch.Size([1, 2, 109, 256])
# torch.Size([1, 2, 109, 256])
# torch.Size([1, 2, 55, 128])
# torch.Size([1, 2, 28, 64])
# torch.Size([1, 2, 14, 32])
# torch.Size([1, 2, 7, 16])

#%%
# flow_truth = read_flow(join(sintel_flow_gt, scene, flowseq[imgi]))
#
# flow = flow.data.cpu()
# flow_up = F.interpolate(flow, (flow.shape[-2] * 4, flow.shape[-1] * 4), mode='bilinear')
# flow = flow[0].numpy().transpose((1, 2, 0))
# flow_up = flow_up[0].numpy().transpose((1, 2, 0))  # zoom(flow, [4, 4, 1], order=1)
# flow_im = flow_to_image(flow, display=True)
# flow_up_im = flow_to_image(flow_up, display=True)

# write_flow(flow_up, join(S_clean_output, scene, imgseq[imgi].split(".")[0] + ".flo"))
# imwrite(join(S_clean_output, scene, imgseq[imgi].split(".")[0] + "_flow.png"), flow_im)
# imwrite(join(S_clean_output, scene, imgseq[imgi].split(".")[0] + "_flow_up.png"), flow_up_im)
