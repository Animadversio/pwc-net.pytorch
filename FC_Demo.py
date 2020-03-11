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
FC_output = r"C:\Users\PonceLab\Documents\PWC_net_Binxu\Datasets\FlyingChairs\FlyingChairs_release\output"
imgseq = sorted([fn for fn in listdir(join(FC_path)) if ".ppm" in fn])
for imgi in range(1, len(imgseq) // 2 + 1):
    im1 = cv2.imread(join(FC_path, "%05d_img1.ppm" % imgi))
    im2 = cv2.imread(join(FC_path, "%05d_img2.ppm" % imgi))
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
    write_flow(flow, join(FC_output, "%05d_flow.flo" % imgi))
    imwrite(join(FC_output, "%05d_flow.png" % imgi), flow_im)
# Visualization
# import matplotlib.pyplot as plt
# plt.imshow(flow_im)
# plt.show()