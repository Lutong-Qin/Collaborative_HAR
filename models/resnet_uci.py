import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.slimmable_ops import USBatchNorm2d, USConv2d, USLinear, make_divisible
from utils.config import FLAGS
from utils.model_profiling import model_profiling


class BasicBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding):
        super(BasicBlock, self).__init__()
        self.layer1 = nn.Sequential(
            USConv2d(input_channel, output_channel, kernel_size, stride, padding,bias=False, ratio=[1, 1]),
            USBatchNorm2d(output_channel, ratio=1),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            USConv2d(output_channel, output_channel, (3,1), 1, (1,0),bias=False, ratio=[1, 1]),
            USBatchNorm2d(output_channel, ratio=1),
        )
        self.shortcut = nn.Sequential(
            USConv2d(input_channel, output_channel, kernel_size, stride, padding,bias=False, ratio=[1, 1]),
            USBatchNorm2d(output_channel, ratio=1),
        )
    def forward(self, x):
        identity = self.shortcut(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = x + identity
        x = F.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(ResNet, self).__init__()

        self.layer1 = self._make_layers(input_channel, 64, (6, 1), (3, 1), (1, 0))
        self.layer2 = self._make_layers(64, 128, (6, 1), (3, 1), (1, 0))
        self.layer3 = self._make_layers(128, 256, (6, 1), (3, 1), (1, 0))
        self.fc = USLinear(256*6*9, num_classes, us=[True, False],previous_channel=256)
        self.pool = nn.AdaptiveMaxPool2d((6, 9))

    def _make_layers(self, input_channel, output_channel, kernel_size, stride, padding):
        return BasicBlock(input_channel, output_channel, kernel_size, stride, padding)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        # print('aa',x.shape)
        # x = F.max_pool2d(x, (6,1))
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        return out


def profiling(model, use_cuda):
    """profiling on either gpu or cpu"""
    print('Start model profiling, use_cuda:{}.'.format(use_cuda))
    for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
        model.apply(
            lambda m: setattr(m, 'width_mult', width_mult))
        print('Model profiling with width mult {}x:'.format(width_mult))
        verbose = width_mult == max(FLAGS.width_mult_list)

        model_profiling(
            model, FLAGS.image_size[0], FLAGS.image_size[1],
            verbose=True)

if __name__=='__main__':
    x=torch.rand((50,1,200,3))
    model=CNN(1,6)
    profiling(model, use_cuda=True)
    x = x.cuda()
    y=model(x)
    print(y.shape)
