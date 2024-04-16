import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import torch.nn.functional as Func
import random



class InputList_3(object):
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, img):
        assert img.size()[1] == self.scales[0][0], 'sensor shape should be equal to max scale'
        input_list = []
        img = img[np.newaxis, :]
        for i in range(len(self.scales)):
            resized_img = Func.interpolate(img, (self.scales[i][0], self.scales[i][1]), mode='nearest-exact')  # [1,3,32,32] [1,3,28,28]  [1,3,24,24] [1,3,20,20]
            resized_img = torch.squeeze(resized_img,0)
            input_list.append(resized_img)

        return input_list