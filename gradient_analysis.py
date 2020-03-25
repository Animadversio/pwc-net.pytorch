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
print("Extractor, lvl1")
for param in pwc.moduleExtractor.moduleOne.parameters():
    print("%s : w %.4E, grad %.4E" % (str(param.shape), param.abs().mean(), param.grad.abs().mean()))
print("Extractor, lvl2")
for param in pwc.moduleExtractor.moduleTwo.parameters():
    print("%s : w %.4E, grad %.4E" % (str(param.shape), param.abs().mean(), param.grad.abs().mean()))
print("Extractor, lvl3")
for param in pwc.moduleExtractor.moduleThr.parameters():
    print("%s : w %.4E, grad %.4E" % (str(param.shape), param.abs().mean(), param.grad.abs().mean()))
print("Extractor, lvl4")
for param in pwc.moduleExtractor.moduleFou.parameters():
    print("%s : w %.4E, grad %.4E" % (str(param.shape), param.abs().mean(), param.grad.abs().mean()))
print("Extractor, lvl5")
for param in pwc.moduleExtractor.moduleFiv.parameters():
    print("%s : w %.4E, grad %.4E" % (str(param.shape), param.abs().mean(), param.grad.abs().mean()))
print("Extractor, lvl6")
for param in pwc.moduleExtractor.moduleSix.parameters():
    print("%s : w %.4E, grad %.4E" % (str(param.shape), param.abs().mean(), param.grad.abs().mean()))
#%%
for submod in pwc.children():
    name = submod._get_name()
    print(name)
    for subsubmod in submod.children():
        subname = subsubmod._get_name()
        print('\t', subname)
        for param in subsubmod.parameters():
            if param.grad is not None:
                print("\t\t%s : w %.4E, grad %.4E" % (str(list(param.shape)), param.abs().mean(), param.grad.abs().mean()))
            else:
                print("\t\t%s : w %.4E, grad not reached" % (str(list(param.shape)), param.abs().mean()))
# for param in pwc.moduleSix.parameters():
#     print("%s : w %.4E, grad %.4E" % (str(param.shape), param.abs().mean(), param.grad.abs().mean()))
#%%
for name, param in pwc.moduleExtractor.moduleFou.parameters().items():
    print("%s %s : %E, %E" % (name, str(list(param.shape)), param.abs().mean()))
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
if __name__ == "__main__":
    pwc = PWC_Net(model_path='models/chairs-things.pytorch').cuda()
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
    # loss_lvl = [0] * 6
    # loss = alpha_w[1] * loss_fun(FLOW_SCALE * predflow_pyr[0] - trflow_pyr[0])
    # loss_lvl[1] = torch.tensor(loss.detach(), requires_grad=False)
    # # loss = torch.tensor(loss_lvl[1], requires_grad=True) # create a new tensor with same grad
    # for level in range(2, 6):
    #     loss_lvl[level] = alpha_w[level] * loss_fun(FLOW_SCALE * predflow_pyr[level] - trflow_pyr[level - 1])
    #     loss += loss_lvl[level]
    loss.backward()
    weight_grad_stats(pwc)