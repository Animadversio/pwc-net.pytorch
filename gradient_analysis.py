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
SintelClean = SintelDataset(render='clean', cropsize=None, torchify=True)

#%%
def loss_fun(diff):  # EPE loss for each pixel summed over space mean over samples
    return torch.mean(torch.sum(torch.sqrt(torch.sum(torch.pow(diff, 2), dim=1)), dim=[1, 2])).squeeze()
FLOW_SCALE = 20.0
#%%
optimizer = optim.SGD(pwc.parameters(), lr=0.00001, )  # weight_decay=0.0004)
alpha_w = [None, 0.005, 0.01, 0.02, 0.08, 0.32]
optimizer.zero_grad()
im1, im2, trflow = SintelClean[100]
trflow_pyr = resize_pyramid(trflow.unsqueeze(0).cuda())
predflow_pyr = pwc(im1.unsqueeze(0).cuda(), im2.unsqueeze(0).cuda())
loss_lvl = [0] * 6
loss = alpha_w[1] * loss_fun(FLOW_SCALE * predflow_pyr[0] - trflow_pyr[0])
loss_lvl[1] = torch.tensor(loss.detach(), requires_grad=False)
# loss = torch.tensor(loss_lvl[1], requires_grad=True) # create a new tensor with same grad
for level in range(2, 6):
    loss_lvl[level] = alpha_w[level] * loss_fun(FLOW_SCALE * predflow_pyr[level] - trflow_pyr[level - 1])
    loss += loss_lvl[level]
loss.backward()
#%%
def weight_grad_stats(model):
    for name, submod in model.named_children():
        classname = submod._get_name()
        print(name, classname)
        for subname, subsubmod in submod.named_children():
            subclassname = subsubmod._get_name()
            print('\t', subname, subclassname)
            for param in subsubmod.parameters():
                if param.grad is not None:
                    print("\t\t%s : w %.4E, grad %.4E" % (
                    str(list(param.shape)), param.abs().mean(), param.grad.abs().mean()))
                else:
                    print("\t\t%s : w %.4E, grad not reached" % (str(list(param.shape)), param.abs().mean()))
    return
#%%
def weight_grad_stats_for_plot(model):
    stat_dict = []
    for name, submod in model.named_children():
        classname = submod._get_name()
        print(name, classname)
        for subname, subsubmod in submod.named_children():
            subclassname = subsubmod._get_name()
            print('\t', subname, subclassname)
            for param in subsubmod.parameters():
                if param.grad is not None:
                    print("\t\t%s : w %.4E, grad %.4E" % (
                    str(list(param.shape)), param.abs().mean(), param.grad.abs().mean()))
                    stat_dict.append((name, subname, param.abs().mean().detach().cpu().numpy(), param.grad.abs().mean().detach().cpu().numpy()))
                else:
                    print("\t\t%s : w %.4E, grad not reached" % (str(list(param.shape)), param.abs().mean()))
    return stat_dict
#%% Counting number of parameters
param_num = 0
for param in pwc.moduleSix.parameters():
    # print(np.prod(list(param.shape)))
    param_num += np.prod(list(param.shape))
print(param_num)
#%%
# pwc = PWC_Net(model_path='models/chairs-things.pytorch').cuda()
pwc = PWC_Net(model_path='../PWC-Net/pwc_net.pth.tar').cuda()
for param in pwc.moduleExtractor.moduleSix.parameters():
    if param.dim() > 1:
        torch.nn.init.xavier_normal_(param)
optimizer = optim.SGD(pwc.parameters(), lr=0.00001, )  # weight_decay=0.0004)
alpha_w = [None, 0.005, 0.01, 0.02, 0.08, 0.32]
optimizer.zero_grad()
im1, im2, trflow = SintelClean[101]
trflow_pyr = resize_pyramid(trflow.unsqueeze(0).cuda())
predflow_pyr = pwc(im1.unsqueeze(0).cuda(), im2.unsqueeze(0).cuda())
loss = loss_fun(FLOW_SCALE * predflow_pyr[5] - trflow_pyr[5 - 1])
loss_lvl = [0] * 6
# loss = alpha_w[1] * loss_fun(FLOW_SCALE * predflow_pyr[0] - trflow_pyr[0])
# loss_lvl[1] = torch.tensor(loss.detach(), requires_grad=False)
# # loss = torch.tensor(loss_lvl[1], requires_grad=True) # create a new tensor with same grad
# for level in range(2, 6):
#     loss_lvl[level] = alpha_w[level] * loss_fun(FLOW_SCALE * predflow_pyr[level] - trflow_pyr[level - 1])
#     loss += loss_lvl[level]
loss.backward()
weight_grad_stats(pwc)
#%% Inspecting Weights of the NVLab version
NVlab_st_dict = torch.load('../PWC-Net/pwc_net.pth.tar')
for name, weight in NVlab_st_dict.items():
    print("%s: %s w %.4E" % (name, str(list(weight.shape)), weight.abs().mean()))
#%%
from torchsummary import summary  # need to support multiple input!
#%%
from visualization import visualize_grad_weight
pwc = PWC_Net(model_path="../pytorch-pwc/network-default.pytorch").cuda()
#%%
im1, im2, trflow = SintelClean[100]
trflow_pyr = resize_pyramid(trflow.unsqueeze(0).cuda())
predflow_pyr = pwc(im1.unsqueeze(0).cuda(), im2.unsqueeze(0).cuda())
alpha_w = [None, 0.005, 0.01, 0.02, 0.08, 0.32]
loss = alpha_w[1] * loss_fun(FLOW_SCALE * predflow_pyr[0] - trflow_pyr[0])
for level in range(2, 6):
    loss += alpha_w[level] * loss_fun(FLOW_SCALE * predflow_pyr[level] - trflow_pyr[level - 1])
loss.backward()
# weight_grad_stats(pwc)
stat_dict = weight_grad_stats_for_plot(pwc)
figh = visualize_grad_weight(stat_dict)
#%%
pwc = PWC_Net(model_path="../pytorch-pwc/network-default.pytorch").cuda()

#%% Plot The Weights and gradient scale through scatter
import matplotlib
namelvl = ['moduleOne', 'moduleTwo', 'moduleThr', 'moduleFou', 'moduleFiv', 'moduleSix']
plt.figure()
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
matplotlib.rc('font', size=14)
plt.rc('legend', fontsize=14)    # legend fontsize
for lvl in range(5):
    meanwg = np.array([[meanw, meang] for name, subname, meanw, meang in stat_dict if (name is 'moduleExtractor' and subname is namelvl[lvl])])
    plt.scatter(np.log10(meanwg[:,0]), np.log10(meanwg[:,1]), c=plt.cm.jet(lvl/5.0), alpha=0.5, label="Extractor Lvl %d"%(lvl+1))
    if lvl >= 1:
        DSmeanwg = np.array([[meanw, meang] for name, subname, meanw, meang in stat_dict if
                           (name is namelvl[lvl])])
        plt.scatter(np.log10(DSmeanwg[:, 0]), np.log10(DSmeanwg[:, 1]), c=plt.cm.jet(lvl / 5.0), alpha=0.5, label="Decoder Lvl %d" % (lvl + 1), marker='*')
meanwg = np.array([[meanw, meang] for name, subname, meanw, meang in stat_dict if (name is 'moduleRefiner')])
plt.scatter(np.log10(meanwg[:,0]), np.log10(meanwg[:,1]), c=plt.cm.jet(1.0/5.0), alpha=0.5, label="Refiner", marker="s")
plt.xlabel("mean amplitude of weight (log)")
plt.ylabel("mean amplitude of gradient (log)")
plt.title("")
plt.legend()
plt.show()
#%% Activation of the Feature Tensor and Cost Volume
pwc.debug = True
pwc.eval()
predflow_pyr, featPyr1, featPyr2 = pwc(im1.unsqueeze(0).cuda(), im2.unsqueeze(0).cuda())
for lvl in range(5,0,-1):
    corrVol = pwc.moduleSix.moduleCorrelation(featPyr1[lvl], featPyr2[lvl])
    print("FeatVol %s, act %.4E \tCorrelVolume %s, act %.4E" %(str(list(featPyr1[lvl].shape)), featPyr1[lvl].abs().mean(), str(list(corrVol.shape)), corrVol.abs().mean()))

#%%
#%%
# print("Extractor, lvl1")
# for param in pwc.moduleExtractor.moduleOne.parameters():
#     print("%s : w %.4E, grad %.4E" % (str(param.shape), param.abs().mean(), param.grad.abs().mean()))
# print("Extractor, lvl2")
# for param in pwc.moduleExtractor.moduleTwo.parameters():
#     print("%s : w %.4E, grad %.4E" % (str(param.shape), param.abs().mean(), param.grad.abs().mean()))
# print("Extractor, lvl3")
# for param in pwc.moduleExtractor.moduleThr.parameters():
#     print("%s : w %.4E, grad %.4E" % (str(param.shape), param.abs().mean(), param.grad.abs().mean()))
# print("Extractor, lvl4")
# for param in pwc.moduleExtractor.moduleFou.parameters():
#     print("%s : w %.4E, grad %.4E" % (str(param.shape), param.abs().mean(), param.grad.abs().mean()))
# print("Extractor, lvl5")
# for param in pwc.moduleExtractor.moduleFiv.parameters():
#     print("%s : w %.4E, grad %.4E" % (str(param.shape), param.abs().mean(), param.grad.abs().mean()))
# print("Extractor, lvl6")
# for param in pwc.moduleExtractor.moduleSix.parameters():
#     print("%s : w %.4E, grad %.4E" % (str(param.shape), param.abs().mean(), param.grad.abs().mean()))
# #%%
# for submod in pwc.children():
#     name = submod._get_name()
#     print(name)
#     for subsubmod in submod.children():
#         subname = subsubmod._get_name()
#         print('\t', subname)
#         for param in subsubmod.parameters():
#             if param.grad is not None:
#                 print("\t\t%s : w %.4E, grad %.4E" % (str(list(param.shape)), param.abs().mean(), param.grad.abs().mean()))
#             else:
#                 print("\t\t%s : w %.4E, grad not reached" % (str(list(param.shape)), param.abs().mean()))
# # for param in pwc.moduleSix.parameters():
# #     print("%s : w %.4E, grad %.4E" % (str(param.shape), param.abs().mean(), param.grad.abs().mean()))
# #%%
# for name, param in pwc.moduleExtractor.moduleFou.parameters().items():
#     print("%s %s : %E, %E" % (name, str(list(param.shape)), param.abs().mean()))