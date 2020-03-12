import cv2
import numpy as np

import torch
import torch.nn.functional as F
from PWC_src import PWC_Net
from PWC_src import flow_to_image, read_flow, write_flow, flow_error, segment_flow

FLOW_SCALE = 20.0
# Build model
pwc = PWC_Net(model_path='models/sintel.pytorch')
# pwc = PWC_Net(model_path='models/chairs-things.pytorch')
pwc = pwc.cuda()
pwc.eval()
#%% Inference Training set
from os import listdir, makedirs
from os.path import join
from imageio import imwrite
from scipy.ndimage import zoom
import time
# S_clean_path = r"C:\Users\PonceLab\Documents\PWC_net_Binxu\Datasets\Sintel\training\clean"
# S_clean_output = r"C:\Users\PonceLab\Documents\PWC_net_Binxu\Datasets\Sintel\training_output"
S_clean_path = r"C:\Users\PonceLab\Documents\PWC_net_Binxu\Datasets\Sintel\test\clean"
S_clean_output = r"C:\Users\PonceLab\Documents\PWC_net_Binxu\Datasets\Sintel\test_output"
scenes = listdir(S_clean_path)
for scene in scenes:
    imgseq = sorted(listdir(join(S_clean_path, scene)))
    makedirs(join(S_clean_output, scene), exist_ok=True)
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
        flow_up = F.interpolate(flow, (flow.shape[-2]*4, flow.shape[-1]*4), mode='bilinear')
        flow = flow[0].numpy().transpose((1,2,0))
        flow_up = flow_up[0].numpy().transpose((1, 2, 0)) # zoom(flow, [4, 4, 1], order=1)
        flow_im = flow_to_image(flow, display=True)
        flow_up_im = flow_to_image(flow_up, display=True)
        write_flow(flow_up, join(S_clean_output, scene, imgseq[imgi].split(".")[0] + ".flo"))
        imwrite(join(S_clean_output, scene, imgseq[imgi].split(".")[0] + "_flow.png"), flow_im)
        imwrite(join(S_clean_output, scene, imgseq[imgi].split(".")[0] + "_flow_up.png"), flow_up_im)
#%%
sintel_flow_gt = r"C:\Users\PonceLab\Documents\PWC_net_Binxu\Datasets\Sintel\training\flow"
for scene in scenes[0:1]:
    flowseq = sorted(listdir(join(sintel_flow_gt, scene)))
    for imgi in range(len(flowseq)):
        flow_arr = read_flow(join(sintel_flow_gt, scene, flowseq[imgi]))
#%%
from scipy.ndimage import zoom
err_col = {}
err_all = []
for scene in scenes:
    print("Scene %s" % scene)
    flowseq = sorted([fn for fn in listdir(join(S_clean_output, scene)) if ".flo" in fn])
    err_col[scene] = []
    for imgi in range(len(flowseq)):
        flow_infer = read_flow(join(S_clean_output, scene, flowseq[imgi]))
        flow_truth = read_flow(join(sintel_flow_gt, scene, flowseq[imgi]))
        flow_upsamp = zoom(flow_infer, [4, 4, 1], order=1)
        mepe = flow_error(flow_truth[:, :, 0], flow_truth[:, :, 1], flow_upsamp[:, :, 0], flow_upsamp[:, :, 1])
        print("%.3f"%mepe)
        err_all.append(mepe)
        err_col[scene].append(mepe)
import pickle
pickle.dump({"err_all":err_all, "err_dict":err_col}, open("..\\Sintel_clean_EPE.pk", "wb"))
#%%
S_clean_output = r"C:\Users\PonceLab\Documents\PWC_net_Binxu\Datasets\Sintel\test_output"
sintel_flow_gt_test = r"C:\Users\PonceLab\Documents\PWC_net_Binxu\Datasets\Sintel\test\clean"
test_err_col = {}
test_err_all = []
for scene in scenes:
    print("Scene %s" % scene)
    flowseq = sorted([fn for fn in listdir(join(S_clean_output, scene)) if ".flo" in fn])
    test_err_col[scene] = []
    for imgi in range(len(flowseq)):
        flow_infer = read_flow(join(S_clean_output, scene, flowseq[imgi]))
        flow_truth = read_flow(join(sintel_flow_gt_test, scene, flowseq[imgi]))
        # flow_upsamp = zoom(flow_infer, [4, 4, 1], order=1)
        mepe = flow_error(flow_truth[:, :, 0], flow_truth[:, :, 1], flow_infer[:, :, 0], flow_infer[:, :, 1])
        print("%.3f"%mepe)
        err_all.append(mepe)
        test_err_col[scene].append(mepe)
import pickle
pickle.dump({"err_all":test_err_all, "err_dict":test_err_col}, open("..\\Sintel_clean_test_EPE.pk", "wb"))
#%%
D = pickle.load(open("..\\Sintel_clean_EPE.pk", "rb"))
#%%
from scipy.ndimage import zoom
flow_upsamp = zoom(flow_arr, [4, 4, 1], order=1)
#%% Visualization
import matplotlib.pyplot as plt
plt.imshow(flow_to_image(flow_upsamp))
plt.show()

#%%
plt.figure()
plt.hist(err_all, 100)
YLIM = plt.ylim()
plt.vlines(np.mean(err_all), YLIM[0], YLIM[1], colors="b")
plt.vlines(np.median(err_all), YLIM[0], YLIM[1], colors="r")
plt.title("Sintel clean training set mean epe\n mean %.2f median %.2f"%(np.mean(err_all), np.median(err_all)))
plt.xlabel("MEPE")
plt.ylabel("frame pairs")
plt.savefig("..\\Results\\sintel_clean_train_epe.png")
plt.show()