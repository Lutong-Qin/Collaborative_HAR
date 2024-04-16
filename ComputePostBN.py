import numpy as np
import torch
import torch.nn.functional as F
from models.slimmable_ops import *


def adjust_bn_layers(module):
    if isinstance(module, nn.BatchNorm2d):
        module.reset_running_stats()

        module._old_momentum = module.momentum

        module.momentum = 0.1

        module._old_training = module.training
        module._old_track_running_stats = module.track_running_stats

        module.training = True
        module.track_running_stats = True


def restore_original_settings_of_bn_layers(module):
    if isinstance(module, nn.BatchNorm2d):
        module.momentum = module._old_momentum
        module.training = module._old_training
        module.track_running_stats = module._old_track_running_stats


def adjust_momentum(module, t):
    if isinstance(module, nn.BatchNorm2d):
        module.momentum = 1 / (t + 1)


def ComputeBN(net, postloader, resolution, num_batch=8):
    net.train()

    net.apply(adjust_bn_layers)

    with torch.no_grad():

        for batch_idx, (inputs, targets) in enumerate(postloader):
            img = inputs[0].cuda()
            net.apply(lambda m: adjust_momentum(m, batch_idx))

            _ = net(F.interpolate(img, (resolution[0], resolution[1]), mode='nearest-exact'))
            if not batch_idx < num_batch:
                break
    net.apply(restore_original_settings_of_bn_layers)

    net.eval()

    return net
