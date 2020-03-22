#!/usr/bin/env python
#%%
import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .correlation_package import correlation
from .correlation import correlation

class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()

        self.moduleOne = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleTwo = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleThr = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFou = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFiv = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleSix = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

    def forward(self, tensorInput):
        tensorOne = self.moduleOne(tensorInput)
        tensorTwo = self.moduleTwo(tensorOne)
        tensorThr = self.moduleThr(tensorTwo)
        tensorFou = self.moduleFou(tensorThr)
        tensorFiv = self.moduleFiv(tensorFou)
        tensorSix = self.moduleSix(tensorFiv)
        # Passing the image through a series of conv layer modules
        # Use the feature tensors to fed the correlation and processing layers.
        return [ tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv, tensorSix ]


class Backward(nn.Module):
    # The so-called Sptial Transformming Flow part. 
    # Use the tensorFlow to warp the tensorInput tensor. 
    def __init__(self):
        super(Backward, self).__init__()

    def forward(self, tensorInput, tensorFlow):
        if hasattr(self, 'tensorPartial') == False or self.tensorPartial.size(0) != tensorFlow.size(0) or self.tensorPartial.size(2) != tensorFlow.size(2) or self.tensorPartial.size(3) != tensorFlow.size(3):
            self.tensorPartial = tensorFlow.new_ones(tensorFlow.size(0), 1, tensorInput.size(2), tensorInput.size(3))
            # Attached to the last layer of input to be warpped as mask!
        if hasattr(self, 'tensorGrid') == False or self.tensorGrid.size(0) != tensorFlow.size(0) or self.tensorGrid.size(2) != tensorFlow.size(2) or self.tensorGrid.size(3) != tensorFlow.size(3):
            tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
            tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
            self.tensorGrid = torch.cat([tensorHorizontal, tensorVertical], 1).cuda()

        tensorInput = torch.cat([tensorInput, self.tensorPartial], 1)
        tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :]/((tensorInput.size(3)-1.0)/2.0),
                                tensorFlow[:, 1:2, :, :]/((tensorInput.size(2)-1.0)/2.0)], 1)

        tensorOutput = F.grid_sample(input=tensorInput, grid=(self.tensorGrid + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)
        tensorMask = tensorOutput[:, -1:, :, :]; tensorMask[tensorMask > 0.999] = 1.0; tensorMask[tensorMask < 1.0] = 0.0

        return tensorOutput[:, :-1, :, :] * tensorMask


class Decoder(nn.Module):
    def __init__(self, intLevel):
        super(Decoder, self).__init__()

        intPrevious = [None, None, 81+32+2+2, 81+64+2+2, 81+96+2+2, 81+128+2+2, 81, None][intLevel+1]
        intCurrent = [None, None, 81+32+2+2, 81+64+2+2, 81+96+2+2, 81+128+2+2, 81, None][intLevel+0]
        # Use deconvolution to do up-sampling 
        if intLevel < 6: self.moduleUpflow = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
        if intLevel < 6: self.moduleUpfeat = nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)
        # upsample while compress the feature to 2d. 
        # Backward function conduct the spatial warping using the flow 
        if intLevel < 6: self.dblBackward = [None, None, None, 5.0, 2.5, 1.25, 0.625, None ][intLevel+1]
        if intLevel < 6: self.moduleBackward = Backward() # Use the flow field to warp the feature tensor. 

        self.moduleCorrelation = correlation.FunctionCorrelation  # correlation.Correlation() #
        self.moduleCorreleaky = nn.LeakyReLU(inplace=False, negative_slope=0.1)
        # Dense Net Architecture, cat the output of all previous modules as input! 
        # Shrink the depth of output channels
        # keep the spatial dimension the same! 
        self.moduleOne = nn.Sequential(
            nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleTwo = nn.Sequential(
            nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleThr = nn.Sequential(
            nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFou = nn.Sequential(
            nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFiv = nn.Sequential(
            nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleSix = nn.Sequential(
            nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1)
        )


    def forward(self, tensorFirst, tensorSecond, objectPrevious):
        tensorFlow = None
        tensorFeat = None

        if objectPrevious is None:  # Top of the pyramid! no need to upsample flow and feature map.
            tensorFlow = None
            tensorFeat = None
            # correlation of 2 tensor as initial rough estimate of cost volume
            tensorVolume = self.moduleCorreleaky(self.moduleCorrelation(tensorFirst, tensorSecond))
            tensorFeat = torch.cat([tensorVolume], 1)  # Leaky ReLU of the correlation between the 2 tensors
            # Only the cost volume goes into DenseNet
        elif objectPrevious is not None:
            # F.pad(, [0, pW, 0, pH])
            tensorFlow = pad_as_size(self.moduleUpflow(objectPrevious['tensorFlow']), output_size=tensorFirst.shape[-2:])
            tensorFeat = pad_as_size(self.moduleUpfeat(objectPrevious['tensorFeat']), output_size=tensorFirst.shape[-2:])
            tensorVolume = self.moduleCorreleaky(self.moduleCorrelation(tensorFirst, self.moduleBackward(tensorSecond, tensorFlow*self.dblBackward).contiguous() ))
            tensorFeat = torch.cat([tensorVolume, tensorFirst, tensorFlow, tensorFeat], 1)
            # Cost Volume, First tensor, Flow map and the compressed Feature tensor from last pyramid level.
            # at Module Four the spatial dimension of tensorFirst and tensorFeat doesn't match
        # DenseNet process the tensorFeat and the `moduleSix` conv that to be a 2 chan tensorFlow
        tensorFeat = torch.cat([self.moduleOne(tensorFeat), tensorFeat], 1)
        tensorFeat = torch.cat([self.moduleTwo(tensorFeat), tensorFeat], 1)
        tensorFeat = torch.cat([self.moduleThr(tensorFeat), tensorFeat], 1)
        tensorFeat = torch.cat([self.moduleFou(tensorFeat), tensorFeat], 1)
        tensorFeat = torch.cat([self.moduleFiv(tensorFeat), tensorFeat], 1)
        tensorFlow = self.moduleSix(tensorFeat)

        return {
            'tensorFlow': tensorFlow,
            'tensorFeat': tensorFeat
        }

def pad_as_size(input, output_size):
    pH = output_size[-2] - input.shape[-2]
    pW = output_size[-1] - input.shape[-1]
    return F.pad(input, [0, pW, 0, pH])

class Refiner(nn.Module):
    def __init__(self):
        super(Refiner, self).__init__()
        # Stress the dilated convolution to increase the RF and provide context info.
        self.moduleMain = nn.Sequential(
            nn.Conv2d(in_channels=81+32+2+2+128+128+96+64+32, out_channels=128, kernel_size=3, stride=1, padding=1,  dilation=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2,  dilation=2),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4,  dilation=4),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8,  dilation=8),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16,  dilation=16),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1,  dilation=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1,  dilation=1)
        )

    def forward(self, tensorInput):
        return self.moduleMain(tensorInput)


class PWC_Net(nn.Module):
    def __init__(self, model_path=None, debug=False):
        super(PWC_Net, self).__init__()
        self.model_path = model_path

        self.moduleExtractor = Extractor()
        self.moduleTwo = Decoder(2)
        self.moduleThr = Decoder(3)
        self.moduleFou = Decoder(4)
        self.moduleFiv = Decoder(5)
        self.moduleSix = Decoder(6)
        self.moduleRefiner = Refiner()
        self.load_state_dict(torch.load(self.model_path))
        self.debug = debug


    def forward(self, tensorFirst, tensorSecond):
        tensorFirst = self.moduleExtractor(tensorFirst)
        tensorSecond = self.moduleExtractor(tensorSecond)

        objectEstimate = self.moduleSix(tensorFirst[-1], tensorSecond[-1], None); flow6 = objectEstimate['tensorFlow']
        objectEstimate = self.moduleFiv(tensorFirst[-2], tensorSecond[-2], objectEstimate); flow5 = objectEstimate['tensorFlow']
        objectEstimate = self.moduleFou(tensorFirst[-3], tensorSecond[-3], objectEstimate); flow4 = objectEstimate['tensorFlow']
        objectEstimate = self.moduleThr(tensorFirst[-4], tensorSecond[-4], objectEstimate); flow3 = objectEstimate['tensorFlow']
        objectEstimate = self.moduleTwo(tensorFirst[-5], tensorSecond[-5], objectEstimate); flow2 = objectEstimate['tensorFlow']
        flow_refined = flow2 + self.moduleRefiner(objectEstimate['tensorFeat'])
        if self.training:
            return [flow_refined, flow2, flow3, flow4, flow5, flow6]
        elif self.debug:
            return [flow_refined, flow2, flow3, flow4, flow5, flow6], tensorFirst, tensorSecond
        else:
            return flow_refined  # objectEstimate['tensorFlow'] + self.moduleRefiner(objectEstimate['tensorFeat'])


if __name__ == '__main__':
    net = PWC_Net(model_path='models/sintel.pytorch')
    net.cuda()

