import cv2
import numpy as np
import matplotlib.pylab as plt
from os import listdir, makedirs
from os.path import join
from imageio import imwrite
from scipy.ndimage import zoom
import time

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch import optim
from PWC_src import PWC_Net
from PWC_src import flow_to_image, read_flow, write_flow, flow_error, segment_flow
from datasets import resize_pyramid, SintelDataset, DataLoader
from visualization import visualize_pyr
SintelClean = SintelDataset(render='clean', cropsize=None, torchify=True)
#%% Load network
pwc = PWC_Net(model_path="../pytorch-pwc/network-default.pytorch").cuda()
pwc.debug = True
pwc.eval()
#%%
sampi = np.random.randint(len(SintelClean))
print(SintelClean.scene_src[sampi])
im1, im2, trflow = SintelClean[sampi]
trflow_pyr = resize_pyramid(trflow.unsqueeze(0).cuda())
predflow_pyr, featPyr1, featPyr2 = pwc(im1.unsqueeze(0).cuda(), im2.unsqueeze(0).cuda())
#%% Proof that the output from lvl6 is independent of input.
samplist = np.random.randint(len(SintelClean), size=10)
flow5_col = []
for sampi in samplist:
    print(SintelClean.scene_src[sampi])
    im1, im2, trflow = SintelClean[sampi]
    trflow_pyr = resize_pyramid(trflow.unsqueeze(0).cuda())
    predflow_pyr, featPyr1, featPyr2 = pwc(im1.unsqueeze(0).cuda(), im2.unsqueeze(0).cuda())
    flow5_col.append(predflow_pyr[5].detach().cpu().numpy())
flow5_col = np.array(flow5_col).squeeze().reshape([len(samplist), -1])
print("mean correlation coefficient %.3f"%np.corrcoef(flow5_col).mean())
np.corrcoef(flow5_col)
#%%
figh = visualize_pyr(predflow_pyr, trflow_pyr, im1=im1, im2=im2, level=5)
#%%
UpFlow4 = pwc.moduleFou.moduleUpflow(predflow_pyr[4])
figh=visualize_flows_cmp([UpFlow4, predflow_pyr[4], predflow_pyr[3]]);figh.show()
#%%
visualize_flows_cmp([predflow_pyr[3], pwc.moduleFou.moduleUpflow(predflow_pyr[4]), predflow_pyr[4]]).show()
#%%
from visualization import  visualize_flows_cmp
#%%
UpFlow4 = pwc.moduleFou.moduleUpflow(predflow_pyr[4])

np.corrcoef(UpFlow4.detach().cpu().numpy().reshape(-1), predflow_pyr[3].detach().cpu().numpy().reshape(-1))[0,1]
#%%
#pwc.moduleFou.moduleUpflow
namelvl = ['moduleOne', 'moduleTwo', 'moduleThr', 'moduleFou', 'moduleFiv', 'moduleSix']
for l in range(1, 5):
    UpFlow = pwc.__getattr__(namelvl[l]).moduleUpflow(predflow_pyr[l+1])
    visualize_flows_cmp([predflow_pyr[l], UpFlow, predflow_pyr[l+1]]).show()
#%%
def pad_as_size(input, output_size):
    pH = output_size[-2] - input.shape[-2]
    pW = output_size[-1] - input.shape[-1]
    return F.pad(input, [0, pW, 0, pH])
#%% compare
l = 1
with torch.no_grad():
    UpFlow = pwc.__getattr__(namelvl[l]).moduleUpflow(predflow_pyr[l+1])
    UpFlow = pad_as_size(UpFlow, [predflow_pyr[l].size(2), predflow_pyr[l].size(3)])
    chan_n = pwc.__getattr__(namelvl[l]).moduleSix[0].in_channels
    padded_H = torch.zeros((1, chan_n, UpFlow.size(2), UpFlow.size(3))).cuda()
    padded_H[:, -4:-2, :, :] = UpFlow
    DirUpFlow = pwc.__getattr__(namelvl[l]).moduleSix(padded_H)
    padded_H = torch.zeros(1)
    DirUpFl_res = (predflow_pyr[l] - DirUpFlow).var().numpy()
    UpFl_res = (predflow_pyr[l] - UpFlow).var().numpy()
    interp_res = (predflow_pyr[l] - F.interpolate(predflow_pyr[l + 1],
            size=[predflow_pyr[l].size(2), predflow_pyr[l].size(3)])).var().numpy()
figh = visualize_flows_cmp([predflow_pyr[l], DirUpFlow, UpFlow, predflow_pyr[l+1]],
                    ["Decoder Out lv%d"%(l+1), "Jump Output", "Upsampling", "Decoder Out lv%d"%(l+2)])
figh.savefig("..\\project1\\FlowIntermPredCmp_Lv%d.png"%(l+1))
figh.show()
#%%
l = 4
with torch.no_grad():
    UpFlow = pwc.__getattr__(namelvl[l]).moduleUpflow(predflow_pyr[l+1])
    UpFlow = pad_as_size(UpFlow, [predflow_pyr[l].size(2), predflow_pyr[l].size(3)])
    chan_n = pwc.__getattr__(namelvl[l]).moduleSix[0].in_channels
    padded_H = torch.zeros((1, chan_n, UpFlow.size(2), UpFlow.size(3))).cuda()
    padded_H[:, -4:-2, :, :] = UpFlow
    DirUpFlow = pwc.__getattr__(namelvl[l]).moduleSix(padded_H)
    padded_H = torch.zeros(1)
    tot_var = predflow_pyr[l].var().cpu().numpy()
    DirUpFl_res = (predflow_pyr[l] - DirUpFlow).var().cpu().numpy()
    UpFl_res = (predflow_pyr[l] - UpFlow).var().cpu().numpy()
    interp_res = (predflow_pyr[l] - F.interpolate(predflow_pyr[l + 1],
            size=[predflow_pyr[l].size(2), predflow_pyr[l].size(3)], mode='bilinear')).var().cpu().numpy()
figh = visualize_flows_cmp([predflow_pyr[l], DirUpFlow, UpFlow, predflow_pyr[l+1]],
                    ["Decoder Out lv%d Var %.2E"%(l+1, tot_var), "Jump Output Res %.2E"%DirUpFl_res, "Upsampling Res %.2E"%UpFl_res, "Decoder Out lv%d Res %.2E"%(l+2, interp_res)])
figh.savefig("..\\project1\\FlowIntermPredCmpRes_Lv%d.png" % (l+1))
figh.show()
#%%
l = 4
with torch.no_grad():
    UpFl_W = pwc.__getattr__(namelvl[l]).moduleUpflow.weight
    DirUp_W = pwc.__getattr__(namelvl[l]).moduleSix[0].weight[:, -4:-2, :, :]
#%%
l = 2
with torch.no_grad():
    UpFl_W = pwc.__getattr__(namelvl[l]).moduleUpflow.weight
    DirUp_W = pwc.__getattr__(namelvl[l]).moduleSix[0].weight[:, -4:-2, :, :]
figh = plt.figure(figsize=[4, 2.5])
plt.subplot(2,4,1)
plt.pcolor(UpFl_W[0,0,:,:].detach().cpu().numpy(), vmin=-.2, vmax=.2)
plt.axis("image");plt.axis("off")
plt.title("U->U")
plt.subplot(2,4,2)
plt.pcolor(UpFl_W[0,1,:,:].detach().cpu().numpy(), vmin=-.2, vmax=.2)
plt.axis("image");plt.axis("off")
plt.title("V->U")
plt.subplot(2,4,3)
plt.pcolor(UpFl_W[1,0,:,:].detach().cpu().numpy(), vmin=-.2, vmax=.2)
plt.axis("image");plt.axis("off")
plt.title("U->V")
plt.subplot(2,4,4)
plt.pcolor(UpFl_W[1,1,:,:].detach().cpu().numpy(), vmin=-.2, vmax=.2)
plt.axis("image");plt.axis("off")
plt.title("V->V")
plt.subplot(2,4,5)
plt.pcolor(DirUp_W[0,0,:,:].detach().cpu().numpy(), vmin=-.15, vmax=.15)
plt.axis("image");plt.axis("off")
plt.title("U->U")
plt.subplot(2,4,6)
plt.pcolor(DirUp_W[0,1,:,:].detach().cpu().numpy(), vmin=-.15, vmax=.15)
plt.axis("image");plt.axis("off")
plt.title("V->U")
plt.subplot(2,4,7)
plt.pcolor(DirUp_W[1,0,:,:].detach().cpu().numpy(), vmin=-.15, vmax=.15)
plt.axis("image");plt.axis("off")
plt.title("U->V")
plt.subplot(2,4,8)
plt.pcolor(DirUp_W[1,1,:,:].detach().cpu().numpy(), vmin=-.15, vmax=.15)
plt.axis("image");plt.axis("off")
plt.title("V->V")
plt.show()
figh.savefig("..\\project1\\FlowUpSampDirker_Lv%d.png" % (l+1))
#%%
figh = visualize_flows_cmp([predflow_pyr[l], DirUpFlow, UpFlow, predflow_pyr[l]-DirUpFlow],
                    ["Decoder Out lv%d"%(l+1), "Jump Output", "Upsampling", "Residue to DirFlow"])
figh.show()
#%%
for l in range(1, 2):
    print("Level %d"%(l+1))
    with torch.no_grad():
        UpFl_W = pwc.__getattr__(namelvl[l]).moduleUpflow.weight
        DirUp_W = pwc.__getattr__(namelvl[l]).moduleSix[0].weight[:, -4:-2, :, :]
        print(UpFl_W.sum(axis=[2,3]).cpu().numpy())
        print(DirUp_W.sum(axis=[2, 3]).cpu().numpy())