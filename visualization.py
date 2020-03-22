

FLOW_SCALE = 20.0
import matplotlib.pylab as plt
from PWC_src import flow_to_image, read_flow, write_flow, flow_error, segment_flow
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')
def visualize_pyr(predflow_pyr, trflow_pyr, im1=None, im2=None, level=None):
    sample_n = trflow_pyr[0].shape[0]
    if level is None:
        coln = 6; padn = 0
        figh_list = []
        for s in range(sample_n):
            figh = plt.figure(figsize=[8, 13])
            if im1 is not None:
                coln = 7
                padn = 2
            if im1 is not None:
                if len(im1.shape) == 4:
                    plt.subplot(coln, 2, 1)
                    plt.imshow(im1[s, :].permute([1, 2, 0]).numpy())
                    plt.axis("off")  # plt.xticks([])
                    plt.subplot(coln, 2, 2)
                    plt.imshow(im2[s, :].permute([1, 2, 0]).numpy())
                    plt.axis("off")  # plt.xticks([])
                else:
                    plt.subplot(coln, 2, 1)
                    plt.imshow(im1.permute([1,2,0]).numpy())
                    plt.axis("off")  #plt.xticks([])
                    plt.subplot(coln, 2, 2)
                    plt.imshow(im2.permute([1,2,0]).numpy())
                    plt.axis("off")  #plt.xticks([])
            for l in range(0, 6):
                flowtr, maxrad = flow_to_image(trflow_pyr[max(l - 1, 0)].detach().cpu().permute([0, 2, 3, 1]).numpy()[s, :], get_lim=True)
                flowimg = flow_to_image(FLOW_SCALE * predflow_pyr[l].detach().cpu().permute([0, 2, 3, 1]).numpy()[s, :], radlim=maxrad)
                plt.subplot(coln, 2, 2 * l + 1 + padn)
                plt.imshow(flowimg)
                plt.axis("off")  # plt.xticks([])
                plt.subplot(coln, 2, 2 * l + 2 + padn)
                plt.imshow(flowtr)
                plt.axis("off")  # plt.xticks([])
            # figh.show()
            figh_list.append(figh)
        return figh_list
    else:
        l = level
        coln = 1
        padn = 0
        for s in range(sample_n):
            flowtr, maxrad = flow_to_image(trflow_pyr[max(l - 1, 0)].detach().cpu().permute([0, 2, 3, 1]).numpy()[s, :],
                                           get_lim=True)
            flowimg = flow_to_image(20 * predflow_pyr[l].detach().cpu().permute([0, 2, 3, 1]).numpy()[s, :],
                                    radlim=maxrad)
            figh = plt.figure(figsize=[8, 4])
            plt.subplot(2, 2, 1)
            plt.imshow(flowimg)
            plt.axis("off")  # plt.xticks([])
            plt.subplot(2, 2, 2)
            plt.imshow(flowtr)
            plt.axis("off")  # plt.xticks([])
            #figh.show()
        return figh
#%%
from misc.montage import build_montages
def visualize_samples(im1, im2, flow):
    flow_img = flow_to_image(flow)
    flow_montage = build_montages([im1, im2, flow_img], (436, 1024,), (2, 2))
    plt.figure(figsize=[10, 5])
    plt.imshow(flow_montage[0])
    plt.axis('off')
    return flow_montage[0]
    # imwrite(join(S_clean_err, scene, "frame_%04d_err%.2f.jpg" % (imgi + 1, err_score[imgi])), flow_montage[0])
def visualize_batch_samples(im1, im2, flow, flow_infer=None):
    for si in range(im1.shape[0]):
        flow_img = flow_to_image(flow[si].permute([1, 2, 0]).data.numpy(), display=True)
        if flow_infer is None:
            flow_montage = build_montages([im1[si].permute([1, 2, 0]).data.numpy(), im2[si].permute([1, 2, 0]).data.numpy(), flow_img], (436, 1024,), (2, 2))
        else:
            flow_infer_img = flow_to_image(flow_infer[si].permute([1, 2, 0]).data.numpy(), display=True)
            flow_montage = build_montages(
                [im1[si].permute([1, 2, 0]).data.numpy(), im2[si].permute([1, 2, 0]).data.numpy(), flow_img, flow_infer_img],
                (436, 1024,), (2, 2))
        plt.figure(figsize=[10, 5])
        plt.imshow(flow_montage[0])
        plt.axis('off')
        plt.show()
    return
