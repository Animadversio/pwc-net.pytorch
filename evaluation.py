import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pylab as plt
import time
from PWC_src import PWC_Net
from PWC_src.flowlib import evaluate_flow, flow_to_image, write_flow
from datasets import SintelDataset, DataLoader, resize_pyramid, FlyingChairDataset
import pickle
FLOW_SCALE = 20.0
smallflow = 0.0
UNKNOWN_FLOW_THRESH = 1e7
def eval_flow(pred_flow, gt_flow):
    epe = ((FLOW_SCALE * pred_flow - gt_flow) ** 2).sum(axis=1).sqrt()
    # mepe = epe.mean(axis=[1, 2])
    msk = (gt_flow.abs() > smallflow).sum(axis=1) > 0 # (gt_flow.abs() > smallflow).sum(axis=1) > 0
    mepe = (epe * msk).sum(axis=[1, 2]) / msk.sum(axis=[1,2])
    # epe = epe[ind2]
    return mepe
#%%
# pwc = PWC_Net(model_path='models/sintel.pytorch').cuda()
# pwc = PWC_Net(model_path='../train_log_L2_sum_err/train_demo_ep062_val2.373.pytorch').cuda()
pwc.train() # pwc.eval()

#%%
def eval_mepe(pwc, eval_loader):
    mepe_frs = []
    mepe_level = [[], [], [], [], [], []]  # [[]]*6 won't work!
    idx = []
    t0 = time.time()
    for Bi, sample in enumerate(eval_loader):
        im1, im2, trflow = sample
        trflow_pyr = resize_pyramid(trflow.cuda())
        with torch.no_grad():
            predflow_pyr = pwc(im1.cuda(), im2.cuda())
        flow_up = F.interpolate(predflow_pyr[0].cpu(), (predflow_pyr[0].shape[-2] * 4, predflow_pyr[0].shape[-1] * 4),
                                mode='bilinear')
        # mepes = []
        # for si in range(im1.shape[0]):
        #     mepe = evaluate_flow(trflow[si,:].permute([1,2,0]).numpy(), 20*flow_up[si,:].permute([1,2,0]).numpy())
        #     mepe_frs.append(mepe)
        #     mepes.append(mepe.copy())
        mepes = list(eval_flow(flow_up, trflow).numpy())
        mepe_frs.extend(mepes.copy())
        for level in range(1, 6):
            if level == 1:
                mepes = list(eval_flow(predflow_pyr[0], trflow_pyr[0]).cpu().numpy())
            else:
                mepes = list(eval_flow(predflow_pyr[level], trflow_pyr[level - 1]).cpu().numpy())
            mepe_level[level].extend(mepes.copy())
        idx.extend(list(range(Bi * Bsize, Bi * Bsize + im1.shape[0])))
        if (Bi + 1) % 10 == 0:
            print("%03d %.3f" % (Bi, time.time() - t0))

    print("Full resolution MEPE %.4f (npy)" % np.mean(mepe_frs))
    # print("Full resolution MEPE %.4f (torch)" % np.mean(mepe_frs2))
    for level in range(1, 6):
        print("Level %d MEPE %.4f " % (level + 1, np.mean(mepe_level[level])))
    return mepe_frs, mepe_level
#%%
# pwc = PWC_Net(model_path='../train_log_L2_sum_err/train_demo_ep062_val2.373.pytorch').cuda()
# pwc = PWC_Net(model_path='models/chairs-things.pytorch').cuda()
# pwc = PWC_Net(model_path='models/sintel.pytorch').cuda()
# model ablating the feature relay
from PWC_src.pwc_ablation import PWC_Net as PWC_Net_abl
# pwc = PWC_Net_abl(model_path='../train_log_feat_abl/train_demo_ep064_val2.145.pytorch').cuda()
# model ablating the feature relay and upsampling
from PWC_src.pwc_ablation_ups import PWC_Net as PWC_Net_abl_ups
pwc = PWC_Net_abl_ups("../train_log_FeatUpsAblation/train_demo_ep099_val2.395.pytorch").cuda()
pwc.train()
Bsize = 4
dataSet = SintelDataset(render="clean", torchify=True, cropsize=None)
# dataSet = FlyingChairDataset(torchify=True, cropsize=None)
eval_loader = DataLoader(dataset=dataSet, batch_size=Bsize, shuffle=False, drop_last=False,)
savename = "%s_model_%s" % ("Sintel_clean", "ups_feat_abl_ep99") # feat_abl_ep64
mepe_frs, mepe_level = eval_mepe(pwc, eval_loader)
pickle.dump({"mepe_hrs": mepe_frs, "mepe_level": mepe_level}, open("..\\%s_EPE_lvl.pk" % savename, "wb"))
#%%

#%%
Bsize = 4
SintelClean = SintelDataset(render="clean", torchify=True, cropsize=None)
eval_loader = DataLoader(dataset=SintelClean, batch_size=Bsize, shuffle=False, drop_last=False,)
#%%

# mepe_frs2 = []
mepe_frs = []
mepe_level = [[],[],[],[],[],[]] # [[]]*6 won't work!
idx = []
t0 = time.time()
for Bi, sample in enumerate(eval_loader):
    im1, im2, trflow = sample
    trflow_pyr = resize_pyramid(trflow.cuda())
    with torch.no_grad():
        predflow_pyr = pwc(im1.cuda(), im2.cuda())
    flow_up = F.interpolate(predflow_pyr[0].cpu(), (predflow_pyr[0].shape[-2] * 4, predflow_pyr[0].shape[-1] * 4), mode='bilinear')
    # mepes = []
    # for si in range(im1.shape[0]):
    #     mepe = evaluate_flow(trflow[si,:].permute([1,2,0]).numpy(), 20*flow_up[si,:].permute([1,2,0]).numpy())
    #     mepe_frs.append(mepe)
    #     mepes.append(mepe.copy())
    mepes = list(eval_flow(flow_up, trflow).numpy())
    mepe_frs.extend(mepes.copy())
    for level in range(1, 6):
        if level == 1:
            mepes = list(eval_flow(predflow_pyr[0], trflow_pyr[0]).cpu().numpy())
        else:
            mepes = list(eval_flow(predflow_pyr[level], trflow_pyr[level - 1]).cpu().numpy())
        mepe_level[level].extend(mepes.copy())
    idx.extend(list(range(Bi * Bsize, Bi * Bsize + im1.shape[0])))
    if (Bi+1) % 10 == 0:
        print("%03d %.3f"%(Bi, time.time() - t0))

print("Full resolution MEPE %.4f (npy)" % np.mean(mepe_frs))
# print("Full resolution MEPE %.4f (torch)" % np.mean(mepe_frs2))
for level in range(1, 6):
    print("Level %d MEPE %.4f " % (level+1, np.mean(mepe_level[level])))
#%%
pickle.dump({"mepe_hrs":mepe_frs, "mepe_level":mepe_level}, open("..\\Sintel_clean_EPE_lvl.pk", "wb"))

#%%
im1, im2, trflow = SintelClean[101]

#%%
    # flow_im = flow_to_image(flow, display=True)
    # flow_up_im = flow_to_image(flow_up, display=True)
    # write_flow(flow_up, join(S_clean_output, scene, imgseq[imgi].split(".")[0] + ".flo"))
#%%
prflow = flow_up[si,:].permute([1,2,0]).numpy()
gtflow = trflow[si,:].permute([1,2,0]).numpy()
