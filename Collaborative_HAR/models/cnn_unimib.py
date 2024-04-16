import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.slimmable_ops import USBatchNorm2d, USConv2d, USLinear, make_divisible
from utils.config import FLAGS
from utils.model_profiling import model_profiling
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class CNN(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(CNN, self).__init__()

        self.layer1 = self._make_layers(input_channel, 64, (6, 1), (2, 1), (1, 0))
        self.layer2 = self._make_layers(64, 128, (6, 1), (2, 1), (1, 0))
        self.layer3 = self._make_layers(128, 256, (6, 1), (2, 1), (1, 0))
        # self.fc = nn.Linear(256 * 4 * 9, num_classes)
        self.pool=nn.AdaptiveMaxPool2d((5,3))
        self.fc = USLinear(256 * 5 * 3, num_classes, us=[True, False],previous_channel=256)



    def _make_layers(self, input_channel, output_channel, kernel_size, stride, padding):
        return nn.Sequential(
            USConv2d(input_channel, output_channel, kernel_size,stride , padding, bias=False, ratio=[1, 1]),
            USBatchNorm2d(output_channel, ratio=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        # print('aa',x.shape)
        # x = F.max_pool2d(x, (4,3))
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        # out = F.softmax(out)
        return out



def profiling(model, use_cuda):
    """profiling on either gpu or cpu"""
    print('Start model profiling, use_cuda:{}.'.format(use_cuda))
    for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
        model.apply(
            lambda m: setattr(m, 'width_mult', width_mult))
        print('Model profiling with width mult {}x:'.format(width_mult))
        verbose = width_mult == max(FLAGS.width_mult_list)  #verbose是一个bool值 Ture   #看看寬度是不是最大值

        # b = getattr(FLAGS, 'model_profiling_verbose')#報錯
        # a=getattr(FLAGS, 'model_profiling_verbose', verbose)  #如果model_profiling_verbose不在在，將設置爲verbose  #Ture
        model_profiling(
            model, FLAGS.image_size[0], FLAGS.image_size[1],
            verbose=True)#verbose的作用是 是否打印，原程序是讓最大的網絡寬度打印.如果一直是Ture的話，可以打印縮小後寬度的參數和mac

if __name__=='__main__':
    x=torch.rand((50,1,200,3))
    # x=x.cuda()
    model=CNN(1,6)
    # model.cuda()
    # model.apply(lambda m: setattr(m, 'width_mult', 1))  # 給'''所有'''子模塊包括最大的模塊都加上這個參數
    profiling(model, use_cuda=True)
    x = x.cuda()
    y=model(x)

    print(y.shape)
