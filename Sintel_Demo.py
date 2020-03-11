import cv2
import numpy as np

import torch

from PWC_src import PWC_Net
from PWC_src import flow_to_image, read_flow, write_flow, flow_error, segment_flow

FLOW_SCALE = 20.0
# Build model
pwc = PWC_Net(model_path='models/sintel.pytorch')
# pwc = PWC_Net(model_path='models/chairs-things.pytorch')
pwc = pwc.cuda()
pwc.eval()
#%%
from os import listdir
from os.path import join
from imageio import imwrite
import time
FC_path = r"C:\Users\PonceLab\Documents\PWC_net_Binxu\Datasets\FlyingChairs\FlyingChairs_release\data"
S_clean_path = r"C:\Users\PonceLab\Documents\PWC_net_Binxu\Datasets\Sintel\training\clean"
S_clean_output = r"C:\Users\PonceLab\Documents\PWC_net_Binxu\Datasets\Sintel\training_output"
scenes = listdir(S_clean_path)
for scene in scenes:
    imgseq = sorted(listdir(join(S_clean_path, scene)))
    for imgi in range(len(imgseq) - 1):
        im1 = cv2.imread(join(S_clean_path, scene, imgseq[imgi]))
        im2 = cv2.imread(join(S_clean_path, scene, imgseq[imgi + 1]))
        im1 = torch.from_numpy((im1/255.).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        im2 = torch.from_numpy((im2/255.).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        im1_v = im1.cuda()
        im2_v = im2.cuda()
        start = time.time()
        flow = FLOW_SCALE*pwc(im1_v, im2_v)
        print(time.time()-start)
        flow = flow.data.cpu()
        flow = flow[0].numpy().transpose((1,2,0))
        flow_im = flow_to_image(flow, display=True)
        write_flow(flow, join(S_clean_output, scene, imgseq[imgi].split(".")[0] + ".flo"))
        imwrite(flow_im, join(S_clean_output, scene, imgseq[imgi].split(".")[0] + "_flow.png"))
# Visualization
# import matplotlib.pyplot as plt
# plt.imshow(flow_im)
# plt.show()