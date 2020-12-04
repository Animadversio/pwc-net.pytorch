import cv2
import numpy as np
from os import listdir
from os.path import join
from imageio import imwrite
import time, tqdm
import torch
from PWC_src import PWC_Net
from PWC_src import flow_to_image, read_flow, write_flow, flow_error, segment_flow

FLOW_SCALE = 20.0
# Build model
pwc = PWC_Net(model_path='models/sintel.pytorch')
# pwc = PWC_Net(model_path='models/chairs-things.pytorch')
pwc = pwc.cuda()
pwc.eval()
pwc.requires_grad_(False)
#%%
vid_path = r"N:\Data-WebCam"
flow_output = r"N:\Data-WebCam\flow"
vid = cv2.VideoCapture(join(vid_path,'Video 5.wmv'))
skipframeN=0
gain = 4.0
fri = 0
while(vid.isOpened()):
    ret, frame1 = vid.read()
    for _ in range(skipframeN):
	    ret, _ = vid.read()
    ret, frame2 = vid.read()
    for _ in range(skipframeN):
	    ret, _ = vid.read()
    if ret == False:
        break
    fri += 2*(skipframeN+1)
    im1 = torch.from_numpy((gain*frame1/255.).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    im2 = torch.from_numpy((gain*frame2/255.).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    flow = FLOW_SCALE*pwc(im1.cuda(), im2.cuda())
    flow = flow.data.cpu()
    flow = flow[0].numpy().transpose((1,2,0))
    flow_im = flow_to_image(flow, display=True)
    write_flow(flow, join(flow_output, "%04d_flow.flo" % fri))
    imwrite(join(flow_output, "%04d_flow.png" % fri), flow_im)
    print("Frame %d finish"%fri)
    # break
    # 


imgseq = sorted([fn for fn in listdir(join(FC_path)) if ".ppm" in fn])
for imgi in tqdm(range(1, len(imgseq) // 2 + 1)):
    im1 = cv2.imread(join(FC_path, "%05d_img1.ppm" % imgi))
    im2 = cv2.imread(join(FC_path, "%05d_img2.ppm" % imgi))
    im1 = torch.from_numpy((im1/255.).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    im2 = torch.from_numpy((im2/255.).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    flow = FLOW_SCALE*pwc(im1.cuda(), im2.cuda())
    flow = flow.data.cpu()
    flow = flow[0].numpy().transpose((1,2,0))
    flow_im = flow_to_image(flow, display=True)
    write_flow(flow, join(flow_output, "%05d_flow.flo" % imgi))
    imwrite(join(flow_output, "%05d_flow.png" % imgi), flow_im)