import cv2
import numpy as np
import matplotlib.pylab as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
from PWC_src import PWC_Net
from PWC_src.pwc_ablation_ups import PWC_Net as PWC_Net_abl_ups
from PWC_src.pwc_ablation import PWC_Net as PWC_Net_abl
from PWC_src import flow_to_image, read_flow, write_flow, flow_error, segment_flow, evaluate_flow
from visualization import visualize_pyr, visualize_samples, visualize_batch_samples
FLOW_SCALE = 20.0
import time
from os import listdir, makedirs
from os.path import join
from imageio import imwrite
from scipy.ndimage import zoom
#%
from datasets import SintelDataset, FlyingChairDataset, resize_pyramid, DataLoader
# SintelClean = SintelDataset(render="clean", torchify=True, cropsize=None)
SintelClean_crop = SintelDataset(render="clean", torchify=True, cropsize=(384, 768))
# FCData = FlyingChairDataset(torchify=False)  # , cropsize=(384, 768)
#%# DataLoader
train_n = 800
val_n = 241
Bsize = 4
Stl_train_set, Stl_val_set = torch.utils.data.random_split(SintelClean_crop, [800, 241])
train_loader = DataLoader(dataset=Stl_train_set, batch_size=Bsize,
                    shuffle=True, drop_last=True,)
val_loader = DataLoader(dataset=Stl_val_set, batch_size=Bsize,
                    shuffle=False, drop_last=False,)  # maybe validation doesn't require cropping?
#%%
import matplotlib
matplotlib.use("Agg") # prevent image output
#%%
from PWC_src.pwc_ups_abl import PWC_Net as PWC_Net_ups
# pwc = PWC_Net_abl(model_path='models/chairs-things.pytorch')
pwc = PWC_Net_ups(model_path='models/chairs-things.pytorch')
# pwc = PWC_Net(model_path='models/sintel.pytorch')
# pwc = PWC_Net(model_path='../train_log/train_demo_ep002_val2.144.pytorch')
# pwc = PWC_Net_abl_ups(model_path='models/chairs-things.pytorch')
pwc.cuda()
pwc.train()
#%

#%%
from PWC_src import PWC_Net
pwc = PWC_Net(model_path='../train_log_L2_sum_err/train_demo_ep069_val2.436.pytorch')
pwc.cuda().train()
#%%
def loss_fun(diff, eps=0.01, q=0.4):
    return torch.mean(torch.sum(torch.pow(torch.sum(torch.abs(diff), dim=1) + eps, q), dim=[1, 2])).squeeze()
# def loss_fun(diff):  # EPE loss for each pixel summed over space mean over samples
#     return torch.mean(torch.sum(torch.sqrt(torch.sum(torch.pow(diff, 2), dim=1)), dim=[1, 2])).squeeze()
from torch import optim
from torch.utils.tensorboard import SummaryWriter
# outdir = "..\\train_log_FeatUpsAblation"
# outdir = "..\\train_log_ups_abl" # "..\\train_log_L2_sum_err"
outdir = "..\\train_log_L2_sum_err_cyclic"
writer = SummaryWriter(log_dir=outdir, flush_secs=180)
#optimizer = optim.Adam(pwc.parameters(), lr=0.00001, weight_decay=0.0004)
alpha_w = [None, 0.005, 0.01, 0.02, 0.08, 0.32]
globstep = 0
ep_i = -1
#%%
from torch.optim.lr_scheduler import CyclicLR
optimizer = optim.SGD(pwc.parameters(), lr=0.00001, weight_decay=0.0004)
scheduler = CyclicLR(optimizer, 6.25e-06, 0.00004, step_size_up=2000, step_size_down=2000, mode='triangular2')
# optimizer = optim.Adam(pwc.parameters(), lr=0.000005, weight_decay=0.0004) # change to 5E-6 at 20 epoc train to 70 epoc
# 10 epoc 3E-5, 30 epoc 2E-5, 20 epoc 1E-5, 20 epoc 5E-6, 20 epoc 2.5E-6, 20 epoc 1.25E-6, 20 epoc 1.5E-5
# 20 epoc 1E-5
# optimizer = optim.Adam(pwc.parameters(), lr=0.00001, weight_decay=0.0004)  # starts from 3E-5
epocs = 200
cur_ep = ep_i
for ep_i in range(cur_ep + 1, cur_ep + 1 + epocs):
    t0 = time.time()
    running_loss = 0
    running_mepe = 0
    running_loss_lvl = np.zeros((6,))
    loss_lvl = [0] * 6
    for Bi, sample in enumerate(train_loader):
        # Can I know which image is it?
        optimizer.zero_grad()
        im1, im2, trflow = sample
        trflow_pyr = resize_pyramid(trflow.cuda())
        predflow_pyr = pwc(im1.cuda(), im2.cuda())
        loss_lvl = [0] * 6
        loss = alpha_w[1] * loss_fun(FLOW_SCALE * predflow_pyr[0] - trflow_pyr[0])
        loss_lvl[1] = torch.tensor(loss.detach(), requires_grad=False)
        # loss = torch.tensor(loss_lvl[1], requires_grad=True) # create a new tensor with same grad
        for level in range(2, 6):
            loss_lvl[level] = alpha_w[level] * loss_fun(FLOW_SCALE * predflow_pyr[level] - trflow_pyr[level - 1])
            loss += loss_lvl[level]
        loss.backward()
        optimizer.step()
        scheduler.step()
        globstep += 1
        running_loss += loss.detach().cpu().numpy()
        for level in range(1, 6):
            running_loss_lvl[level] += loss_lvl[level].detach().cpu().numpy()
        mepes = []
        for si in range(im1.shape[0]):
            mepe = evaluate_flow(trflow_pyr[0][si, :].detach().cpu().permute(1, 2, 0).numpy(),
                          FLOW_SCALE * predflow_pyr[0][si, :].detach().cpu().permute(1, 2, 0).numpy())
            running_mepe += mepe
            mepes.append(mepe)
        print("%.3f sec, batch %d, B mepe %.3f, running mepe %.2f, running loss %.2f (%.2f, %.2f, %.2f, %.2f, %.2f)" % (time.time() - t0,
            Bi + 1, np.mean(mepes), running_mepe / (Bi + 1) / Bsize, running_loss / (Bi + 1),
            *tuple(running_loss_lvl[1:] / (Bi + 1))))
        if (Bi) % 50 == 0:
            writer.add_scalar('optim/lr', scheduler.get_lr()[0], global_step=globstep)
            writer.add_scalar('Loss/loss', loss, global_step=globstep)
            writer.add_scalar('Loss/running_loss', running_loss / (Bi + 1), global_step=globstep)
            for l in range(1, 6):
                writer.add_scalar('LossParts/running_loss lvl%d' % l , running_loss_lvl[l] / (Bi + 1), global_step=globstep)
            writer.add_scalar('Eval/mepe', mepe, global_step=globstep)
            writer.add_scalar('Eval/running_mepe', running_mepe / Bsize / (Bi + 1), global_step=globstep)
            figh_list = visualize_pyr(predflow_pyr, trflow_pyr, im1=im1, im2=im2, level=None)
            for si in range(len(figh_list)):
                writer.add_figure('Figure/flow_cmp%d'%si, figh_list[si], global_step=globstep)
    val_loss = 0
    val_mepe = 0
    val_loss_lvl = np.zeros((6,))
    with torch.no_grad():
        for Bi, sample in enumerate(val_loader):
            im1, im2, trflow = sample
            trflow_pyr = resize_pyramid(trflow.cuda())
            predflow_pyr = pwc(im1.cuda(), im2.cuda())
            loss_lvl = [0] * 6
            loss = alpha_w[1] * loss_fun(FLOW_SCALE * predflow_pyr[0] - trflow_pyr[0])
            loss_lvl[1] = torch.tensor(loss.detach(), requires_grad=False)
            for level in range(2, 6):
                loss_lvl[level] = alpha_w[level] * loss_fun(FLOW_SCALE * predflow_pyr[level] - trflow_pyr[level - 1])
                loss += loss_lvl[level]
            val_loss += loss.detach().cpu().numpy()
            for level in range(1, 6):
                val_loss_lvl[level] += loss_lvl[level].detach().cpu().numpy()
            for si in range(im1.shape[0]):
                mepe = evaluate_flow(trflow_pyr[0][si, :].detach().cpu().permute(1, 2, 0).numpy(),
                                     FLOW_SCALE * predflow_pyr[0][si, :].detach().cpu().permute(1, 2, 0).numpy())
                val_mepe += mepe
    writer.add_scalar('optim/lr', scheduler.get_lr()[0], global_step=globstep)
    writer.add_scalar('Loss/val_loss', val_loss / val_n * Bsize, global_step=globstep)
    writer.add_scalar('Eval/val_mepe', val_mepe / val_n, global_step=globstep)
    writer.add_scalar('Loss/full_loss', (running_loss + val_loss) / (val_n + train_n) * Bsize, global_step=globstep)
    for l in range(1, 6):
        writer.add_scalar('LossParts/val_loss lvl%d' % l, val_loss_lvl[l] / val_n * Bsize, global_step=globstep)
        writer.add_scalar('LossParts/full_loss lvl%d' % l, (running_loss_lvl[l] + val_loss_lvl[l]) / (val_n + train_n) * Bsize, global_step=globstep)
    writer.add_scalar('Eval/full_mepe', (running_mepe + val_mepe) / (val_n + train_n), global_step=globstep)
    torch.save(pwc.state_dict(), join(outdir, "train_demo_ep%03d_val%.3f.pytorch" % (ep_i, val_mepe / val_n)))
#%%
# changed loss function to robust loss at 14k steps
#%%
# Using batch size of one to pass, it's fine, nothing wrong.
# But Batch size of 2 will cause distortion in the output! (Batch norm not working?)
#%%
# import matplotlib.pylab as plt
# l = 5
# flowimg1 = flow_to_image(predflow_pyr[l].detach().cpu().permute([0, 2, 3, 1]).numpy()[1, :]*20)
# flowtr1 = flow_to_image(trflow_pyr[max(l-1, 0)].detach().cpu().permute([0, 2, 3, 1]).numpy()[1, :])
# #flowimg2 = flow_to_image(predflow_pyr[l].detach().cpu().permute([0,2,3,1]).numpy()[1, :])
# plt.figure(figsize=[8, 12])
# for l in range(0, 6):
#     flowimg1 = flow_to_image(predflow_pyr[l].detach().cpu().permute([0,2,3,1]).numpy()[1, :])
#     flowtr1 = flow_to_image(trflow_pyr[max(l-1, 0)].detach().cpu().permute([0,2,3,1]).numpy()[1, :])
#     plt.subplot(6, 2, 2 * l + 1)
#     plt.imshow(flowimg1)
#     plt.subplot(6, 2, 2 * l + 2)
#     plt.imshow(flowtr1)
# plt.show()
#
# #%%
# plt.figure(figsize=[10,5])
# plt.imshow(trflow_pyr[max(l, 0)].detach().cpu().permute([0, 2, 3, 1]).numpy()[0, :,:,1]/20)
# plt.show()
# #%%
# torch.save(pwc.state_dict(), r"models/train_demo.pytorch")