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
from os import listdir, makedirs
from os.path import join
from imageio import imwrite
from scipy.ndimage import zoom
import time
Cats_path = r"C:\Users\PonceLab\Documents\PWC_net_Binxu\Datasets\Cats\frames"
output_path = r"C:\Users\PonceLab\Documents\PWC_net_Binxu\Datasets\Cats\outputs"

fr_stride = 8
vid_fns = listdir(Cats_path)
vid_fn = vid_fns[1]#"cat001_01"
for vid_fn in vid_fns[2:]:
    imgseq = sorted([fn for fn in listdir(join(Cats_path, vid_fn)) if "cat" in fn and "L1" in fn])
    makedirs(join(output_path, vid_fn), exist_ok=True)
    for imgi in range(1, len(imgseq)-fr_stride -1, fr_stride):
        start = time.time()
        im1 = cv2.imread(join(Cats_path, vid_fn, "%s_%04d_L1.jpg" % (vid_fn, imgi)))
        im2 = cv2.imread(join(Cats_path, vid_fn, "%s_%04d_L1.jpg" % (vid_fn, imgi + fr_stride)))
        # im2 = cv2.imread(join(Cats_path, "%s_%04d_R1.jpg" % (vid_fn, imgi)))
        im1 = torch.from_numpy((im1/255.).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        im2 = torch.from_numpy((im2/255.).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        im1_v = im1.cuda()
        im2_v = im2.cuda()
        #start = time.time()
        flow = FLOW_SCALE*pwc(im1_v, im2_v)
        print(time.time()-start)
        flow = flow.data.cpu()
        flow = flow[0].numpy().transpose((1,2,0))
        flow_up = zoom(flow, [4, 4, 1], order=1)
        flow_im = flow_to_image(flow_up, display=True)
        # write_flow(flow, join(output_path, "%05d_flow.flo" % imgi))
        imwrite(join(output_path, vid_fn, "%05d_flow_s%d_L1.jpg" % (imgi, fr_stride)), flow_im)
