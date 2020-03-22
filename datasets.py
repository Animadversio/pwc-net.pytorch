from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, RandomVerticalFlip
from torchvision.transforms.functional import crop as Crop
import cv2
import numpy as np
import matplotlib.pylab as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
from os import listdir, makedirs
from os.path import join
from imageio import imwrite
from scipy.ndimage import zoom
import time
from PWC_src.flowlib import *
toTsr = transforms.ToTensor()
# Dataset system
# Training System should handle
# * Data preprocessing
# * Data feeding
# *
#%%
class SintelDataset(Dataset):
    def __init__(self, render="clean", preload=False, torchify=True, cropsize=None):
        if render == "clean":
            self.frame_dir = r"C:\Users\PonceLab\Documents\PWC_net_Binxu\Datasets\Sintel\training\clean"
        elif render == "final":
            self.frame_dir = r"C:\Users\PonceLab\Documents\PWC_net_Binxu\Datasets\Sintel\training\final"
        elif render == "albedo":
            self.frame_dir = r"C:\Users\PonceLab\Documents\PWC_net_Binxu\Datasets\Sintel\training\albedo"
        self.flow_dir = r"C:\Users\PonceLab\Documents\PWC_net_Binxu\Datasets\Sintel\training\flow"
        self.render = render
        self.torchify = torchify
        self.preloaded = False
        self.build_idx(preload)
        if cropsize is None:
            self.rnd_crop = False
        else:
            self.rnd_crop = True
            self.cropper = RandomCrop(cropsize)  # (384, 768)

    def build_idx(self, preload=False):
        scenes = listdir(self.frame_dir)
        self.preloaded = preload
        self.index = []
        self.scene_src = []
        self.dataset = []
        for scene in scenes:
            imgseq = sorted(listdir(join(self.frame_dir, scene)))
            for imgi in range(len(imgseq) - 1):
                im1_path = join(self.frame_dir, scene, imgseq[imgi])
                im2_path = join(self.frame_dir, scene, imgseq[imgi + 1])
                flow_path = join(self.flow_dir, scene, "frame_%04d.flo" % (imgi + 1))
                self.index.append((im1_path, im2_path, flow_path))
                self.scene_src.append((scene, imgi + 1))
                if preload:
                    im1 = cv2.imread(im1_path)
                    im2 = cv2.imread(im2_path)
                    flow = read_flow(flow_path)
                    self.dataset.append((im1, im2, flow))

    def transform(self, im1, im2, flow):
        # assume inputs are tensors!
        if self.rnd_crop:
            self.crop_indices = self.cropper.get_params(im1, output_size=self.cropper.size)
            i, j, h, w = self.crop_indices
            im1 = im1[:, i:i+h, j:j+w] # Crop(im1, i, j, h, w)
            im2 = im2[:, i:i+h, j:j+w] # Crop(im2, i, j, h, w)
            flow = flow[:, i:i+h, j:j+w] # Crop(flow, i, j, h, w)
        return im1, im2, flow

    def __getitem__(self, idx):
        im1_path, im2_path, flow_path = self.index[idx]
        im1 = cv2.imread(im1_path) # note this is uint8 255 scale
        im2 = cv2.imread(im2_path)
        flow = read_flow(flow_path)
        if not self.torchify:
            return im1, im2, flow
        else:
            if not self.rnd_crop:
                return toTsr(im1), toTsr(im2), toTsr(flow)
            else:
                return self.transform(toTsr(im1), toTsr(im2), toTsr(flow))
        # torch.from_numpy((im1/255.).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)

    def __len__(self):
        return len(self.index)

class FlyingChairDataset(Dataset):
    def __init__(self, preload=False, torchify=True, cropsize=None):
        self.frame_dir = r"C:\Users\PonceLab\Documents\PWC_net_Binxu\Datasets\FlyingChairs\FlyingChairs_release\data"
        self.flow_dir = r"C:\Users\PonceLab\Documents\PWC_net_Binxu\Datasets\FlyingChairs\FlyingChairs_release\data"
        self.torchify = torchify
        self.preloaded = False
        self.build_idx(preload)
        if cropsize is None:
            self.rnd_crop = False
        else:
            self.rnd_crop = True
            self.cropper = RandomCrop(cropsize)  # (384, 768)

    def build_idx(self, preload=False):
        self.preloaded = preload
        self.index = []
        self.scene_src = []
        self.dataset = []
        imgseq = sorted([fn for fn in listdir(self.flow_dir) if ".flo" in fn])
        # imgseq = sorted(listdir(join(self.frame_dir)))
        for imgi in range(len(imgseq)):
            im1_path = join(self.frame_dir, "%05d_img1.ppm" % (imgi + 1))
            im2_path = join(self.frame_dir, "%05d_img2.ppm" % (imgi + 1))
            flow_path = join(self.flow_dir, "%05d_flow.flo" % (imgi + 1))
            self.index.append((im1_path, im2_path, flow_path))
            self.scene_src.append((imgi + 1))
            if preload:
                im1 = cv2.imread(im1_path)
                im2 = cv2.imread(im2_path)
                flow = read_flow(flow_path)
                self.dataset.append((im1, im2, flow))

    def transform(self, im1, im2, flow):
        # assume inputs are tensors!
        if self.rnd_crop:
            self.crop_indices = self.cropper.get_params(im1, output_size=self.cropper.size)
            i, j, h, w = self.crop_indices
            im1 = im1[:, i:i+h, j:j+w]  # Crop(im1, i, j, h, w)
            im2 = im2[:, i:i+h, j:j+w]  # Crop(im2, i, j, h, w)
            flow = flow[:, i:i+h, j:j+w]  # Crop(flow, i, j, h, w)
        return im1, im2, flow

    def __getitem__(self, idx):
        im1_path, im2_path, flow_path = self.index[idx]
        im1 = cv2.imread(im1_path) # note this is uint8 255 scale
        im2 = cv2.imread(im2_path)
        flow = read_flow(flow_path)
        if not self.torchify:
            return im1, im2, flow
        else:
            if not self.rnd_crop:
                return toTsr(im1), toTsr(im2), toTsr(flow)
            else:
                return self.transform(toTsr(im1), toTsr(im2), toTsr(flow))
        # torch.from_numpy((im1/255.).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)

    def __len__(self):
        return len(self.index)

#%%
def resize_pyramid(img, level=6, ds=2):
    """ Resize output image to see the intermediate supervised signal. """
    # assuming input is numpy convention image or torch tensor.
    pyramid = []
    img_cur = img
    if torch.is_tensor(img) and len(img.shape) == 3:
        img_cur = img.unsqueeze(0)
    for i in range(level):
        if not torch.is_tensor(img_cur):
            H, W = img_cur.shape[0], img_cur.shape[1]
            img_cur = cv2.resize(img_cur, (np.ceil(W / ds).astype(np.int), np.ceil(H / ds).astype(np.int)), cv2.INTER_LINEAR)
        else:
            H, W = img_cur.shape[-2], img_cur.shape[-1]
            img_cur = F.interpolate(img_cur, size = (np.ceil(H / ds).astype(np.int), np.ceil(W / ds).astype(np.int)), mode='bilinear')#scale_factor=1/ds)
        if not (i == 0 and torch.is_tensor(img)):
            pyramid.append(img_cur) # escape the 1st level, not used in calculation
        # print(img_cur.shape)
    return pyramid
#%%
