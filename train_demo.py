import cv2
import numpy as np
import matplotlib.pylab as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
from PWC_src import PWC_Net
from PWC_src import flow_to_image, read_flow, write_flow, flow_error, segment_flow
#%%
FLOW_SCALE = 20.0
# Build model
# pwc = PWC_Net(model_path='models/sintel.pytorch')
pwc = PWC_Net(model_path='models/chairs-things.pytorch')
pwc = pwc.cuda()
# pwc.eval()
pwc.train()
#%%
from os import listdir, makedirs
from os.path import join
from imageio import imwrite
from scipy.ndimage import zoom
#%%
from datasets import SintelDataset, FlyingChairDataset, resize_pyramid, DataLoader
# SintelClean = SintelDataset(render="clean", torchify=True, cropsize=None)
SintelClean_crop = SintelDataset(render="clean", torchify=True, cropsize=(384, 768))
FCData = FlyingChairDataset(torchify=False)  # , cropsize=(384, 768)
# DataLoader
#%%
pyr = resize_pyramid(flow)
pyr = resize_pyramid(tsrim1)
#%%
train_n = 800
val_n = 241
Bsize = 4
Stl_train_set, Stl_val_set = torch.utils.data.random_split(SintelClean_crop, [800, 241])
train_loader = DataLoader(dataset=Stl_train_set, batch_size=Bsize,
                    shuffle=True, drop_last=True,)
val_loader = DataLoader(dataset=Stl_val_set, batch_size=Bsize,
                    shuffle=False, drop_last=False,) # maybe validation doesn't require cropping?
                    #sampler=torch.utils.data.RandomSampler(SintelClean))
#%%
pwc = PWC_Net(model_path='models/chairs-things.pytorch')
# pwc = PWC_Net(model_path='models/sintel.pytorch')
pwc.train()
pwc.cuda()
#%%
def loss_fun(diff, eps=0.01, q=0.4):
    return torch.mean(torch.pow(torch.sum(torch.abs(diff), dim=1) + eps, q)).squeeze()
from torch import optim
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="..\\train_log", flush_secs=180)

# pwc = PWC_Net(model_path='models/sintel.pytorch')
optimizer = optim.Adam(pwc.parameters(), lr=0.00001, weight_decay=0.0004)
# loader = DataLoader(dataset=SintelClean, batch_size=1,
#                     shuffle=True, drop_last=True,)
                    #sampler=torch.utils.data.RandomSampler(SintelClean))
alpha_w = [None, 0.005, 0.01, 0.02, 0.08, 0.32]
#%%
import matplotlib
matplotlib.use("Agg") # prevent image output
#%%
epocs = 20
#globstep = 0
for ep_i in range(100, 100+epocs):
    t0 = time.time()
    running_loss = 0
    running_mepe = 0
    for Bi, sample in enumerate(train_loader):
        # Can I know which image is it?
        optimizer.zero_grad()
        im1, im2, trflow = sample
        trflow_pyr = resize_pyramid(trflow.cuda())
        predflow_pyr = pwc(im1.cuda(), im2.cuda())
        loss = alpha_w[1] * loss_fun(FLOW_SCALE * predflow_pyr[0] - trflow_pyr[0])
        for level in range(2, 6):
            loss = loss + alpha_w[level] * loss_fun(FLOW_SCALE * predflow_pyr[level] - trflow_pyr[level - 1])
        loss.backward()
        optimizer.step()
        globstep += 1
        running_loss += loss.detach().cpu().numpy()
        mepes = []
        for si in range(im1.shape[0]):
            mepe = flow_error(trflow_pyr[0][si, 0, :, :].detach().cpu().numpy(),
                              trflow_pyr[0][si, 1, :, :].detach().cpu().numpy(),
                              FLOW_SCALE * predflow_pyr[0][si, 0, :, :].detach().cpu().numpy(),
                              FLOW_SCALE * predflow_pyr[0][si, 1, :, :].detach().cpu().numpy())
            running_mepe += mepe
            mepes.append(mepe)
        print(time.time() - t0, " sec, %d batch, B mepe %.3f, running loss %.2f, running mepe %.2f" % (Bi + 1,
                            np.mean(mepes), running_loss / (Bi + 1) / Bsize, running_mepe / (Bi + 1) / Bsize, ))
        if (Bi) % 25 == 0:
            writer.add_scalar('Loss/loss', loss, global_step=globstep)
            writer.add_scalar('Loss/running_loss', running_loss / Bsize/ (Bi+1), global_step=globstep)
            writer.add_scalar('Eval/mepe', mepe, global_step=globstep)
            writer.add_scalar('Eval/running_mepe', running_mepe / Bsize/ (Bi + 1), global_step=globstep)
            # writer.add_images('Flow/Refined', predflow_pyr[0], global_step=Bi)
            # writer.add_images('Flow/Level2', predflow_pyr[1], global_step=Bi)
            # writer.add_images('Flow/Level3', predflow_pyr[2], global_step=Bi)
            # writer.add_images('Flow/Level4', predflow_pyr[3], global_step=Bi)
            # writer.add_images('Flow/Level5', predflow_pyr[4], global_step=Bi)
            figh_list = visualize_pyr(predflow_pyr, trflow_pyr, im1=im1, im2=im2, level=None)
            for si in range(len(figh_list)):
                writer.add_figure('Figure/flow_cmp%d'%si, figh_list[si], global_step=globstep)
    val_loss = 0
    val_mepe = 0
    with torch.no_grad():
        for Bi, sample in enumerate(val_loader):
            im1, im2, trflow = sample
            trflow_pyr = resize_pyramid(trflow.cuda())
            predflow_pyr = pwc(im1.cuda(), im2.cuda())
            loss = alpha_w[1] * loss_fun(FLOW_SCALE * predflow_pyr[0] - trflow_pyr[0])
            for level in range(2, 6):
                loss = loss + alpha_w[level] * loss_fun(FLOW_SCALE * predflow_pyr[level] - trflow_pyr[level - 1])
            val_loss += loss.detach().cpu().numpy()
            for si in range(im1.shape[0]):
                mepe = flow_error(trflow_pyr[0][si, 0, :, :].detach().cpu().numpy(),
                                  trflow_pyr[0][si, 1, :, :].detach().cpu().numpy(),
                                  FLOW_SCALE * predflow_pyr[0][si, 0, :, :].detach().cpu().numpy(),
                                  FLOW_SCALE * predflow_pyr[0][si, 1, :, :].detach().cpu().numpy())
                val_mepe += mepe
    writer.add_scalar('Loss/val_loss', val_loss / val_n, global_step=globstep)
    writer.add_scalar('Eval/val_mepe', val_mepe / val_n, global_step=globstep)
    writer.add_scalar('Loss/full_loss', (running_loss + val_loss) / (val_n + train_n), global_step=globstep)
    writer.add_scalar('Eval/full_mepe', (running_mepe + val_mepe) / (val_n + train_n), global_step=globstep)
    torch.save(pwc.state_dict(), r"../train_log/train_demo_ep%03d_val%.3f.pytorch" % (ep_i, val_mepe / val_n))
#%%

# Using batch size of one to pass, it's fine, nothing wrong.
# But Batch size of 2 will cause distortion in the output! (Batch norm not working?)
#%%
import matplotlib.pylab as plt
l = 5
flowimg1 = flow_to_image(predflow_pyr[l].detach().cpu().permute([0,2,3,1]).numpy()[1, :]*20)
flowtr1 = flow_to_image(trflow_pyr[max(l-1, 0)].detach().cpu().permute([0,2,3,1]).numpy()[1, :])
#flowimg2 = flow_to_image(predflow_pyr[l].detach().cpu().permute([0,2,3,1]).numpy()[1, :])
plt.figure(figsize=[8, 12])
for l in range(0, 6):
    flowimg1 = flow_to_image(predflow_pyr[l].detach().cpu().permute([0,2,3,1]).numpy()[1, :])
    flowtr1 = flow_to_image(trflow_pyr[max(l-1, 0)].detach().cpu().permute([0,2,3,1]).numpy()[1, :])
    plt.subplot(6, 2, 2 * l + 1)
    plt.imshow(flowimg1)
    plt.subplot(6, 2, 2 * l + 2)
    plt.imshow(flowtr1)
plt.show()
#%%
plt.figure(figsize=[10,5])
plt.imshow(trflow_pyr[max(l, 0)].detach().cpu().permute([0, 2, 3, 1]).numpy()[0, :,:,1]/20)
plt.show()
#%%
torch.save(pwc.state_dict(), r"models/train_demo.pytorch")