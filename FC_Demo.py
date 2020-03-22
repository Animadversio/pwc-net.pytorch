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
#%%
#%% Evaluation
from scipy.ndimage import zoom
FC_path = r"C:\Users\PonceLab\Documents\PWC_net_Binxu\Datasets\FlyingChairs\FlyingChairs_release\data"
FC_output = r"C:\Users\PonceLab\Documents\PWC_net_Binxu\Datasets\FlyingChairs\FlyingChairs_release\output"
flowseq = sorted([fn for fn in listdir(join(FC_path)) if ".flo" in fn])
FC_err_all = []
for imgi in range(len(flowseq)):
    flow_infer = read_flow(join(FC_output, flowseq[imgi]))
    flow_truth = read_flow(join(FC_path, flowseq[imgi]))
    flow_upsamp = zoom(flow_infer, [4, 4, 1], order=1)
    mepe = flow_error(flow_truth[:, :, 0], flow_truth[:, :, 1], flow_upsamp[:, :, 0], flow_upsamp[:, :, 1])
    print("%.3f"%mepe)
    FC_err_all.append(mepe)
import pickle
pickle.dump({"err_all":FC_err_all}, open("..\\FC_EPE.pk", "wb"))
#%%
D = pickle.load(open("..\\FC_EPE.pk", "rb"))

#%% Visualization
import matplotlib.pyplot as plt
#%%
plt.figure()
plt.hist(FC_err_all, 100)
YLIM = plt.ylim()
plt.vlines(np.mean(FC_err_all), YLIM[0], YLIM[1], colors="b")
plt.vlines(np.median(FC_err_all), YLIM[0], YLIM[1], colors="r")
plt.title("Flying Chair Dataset mean epe\n mean %.2f median %.2f"%(np.mean(FC_err_all), np.median(FC_err_all)))
plt.xlabel("MEPE")
plt.ylabel("frame pairs")
plt.savefig("..\\Results\\FlyingChair_epe.png")
plt.show()
#%% Error Inspect
import matplotlib
# matplotlib.use("")
import sys
sys.path.append("misc")
from .misc.montage import build_montages
FC_path = r"C:\Users\PonceLab\Documents\PWC_net_Binxu\Datasets\FlyingChairs\FlyingChairs_release\data"
FC_output = r"C:\Users\PonceLab\Documents\PWC_net_Binxu\Datasets\FlyingChairs\FlyingChairs_release\output"
FC_err = r"C:\Users\PonceLab\Documents\PWC_net_Binxu\Datasets\FlyingChairs\err_examine"

err_score = np.array(FC_err_all)
idx = np.argsort(err_score)
err_frames = idx[-100:-1000:-1]
for imgi in err_frames:
    im1 = cv2.imread(join(FC_path, "%05d_img1.ppm" % (imgi+1)))
    im2 = cv2.imread(join(FC_path, "%05d_img2.ppm" % (imgi+1)))
    flow_infer = read_flow(join(FC_output, "%05d_flow.flo" % (imgi+1)))
    flow_truth = read_flow(join(FC_path, "%05d_flow.flo" % (imgi+1)))
    flow_upsamp = zoom(flow_infer, [4, 4, 1], order=1)
    flow_infer_im = flow_to_image(flow_upsamp)
    flow_truth_im = flow_to_image(flow_truth)
    flow_montage = build_montages([im1, im2, flow_infer_im, flow_truth_im], (384, 512, ), (2, 2))
    imwrite(join(FC_err, "%05d_err%.2f.jpg"%(imgi + 1, err_score[imgi])), flow_montage[0])
    # read_flow(join(sintel_flow_gt_test, scene, flowseq[imgi]))
    # flow_infer = read_flow(join(S_clean_output, scene, flowseq[imgi]))
