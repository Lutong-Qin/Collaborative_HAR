import torch.nn as nn
from utils.config import FLAGS
width_mult = FLAGS.width_mult_range[-1]

def make_divisible(v, divisor=4, min_value=1):  # 确保所有的通道数被8整除
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/\
    0344c5503ee55e24f0de7f37336a6e08f10976fd/\
    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """

    '''# 确保所有的通道数被8整除
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:  # 通道数
    :param divisor:  除数 8
    :param min_value:  最小除数
    :return:
    """
    '''
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.  #请确保向下舍入的幅度不超过10%。
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# x=make_divisible(204)
# print(x)
class USConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, depthwise=False, bias=True,
                 us=[True, True], ratio=[1, 1]):    # ratio是幹啥的？    # us應該是是否使用動態寬度
        in_channels_max = in_channels
        out_channels_max = out_channels
        if us[0]:   #   width_mult是全局外定义的，确实是最大的，这个判断是为了把输入输出变成能被8整除的
            in_channels_max = int(make_divisible(
                in_channels
                * width_mult    #   width_mult是最大的宽度
                / ratio[0]) * ratio[0])     # 为什么要/ratio[0]
        if us[1]:
            out_channels_max = int(make_divisible(
                out_channels
                * width_mult
                / ratio[1]) * ratio[1])
        groups = in_channels_max if depthwise else 1
        #   子类把父类的__init__()放到自己的__init__()当中，这样子类就有了父类的__init__()的那些东西。
        super(USConv2d, self).__init__(
            in_channels_max, out_channels_max,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)       #       这个超类初始化(也就是nn.Conv2d初始化的时候)就把缩小的通道啥的初始化了
        self.depthwise = depthwise
        self.in_channels_basic = in_channels
        self.out_channels_basic = out_channels
        self.width_mult = None
        self.us = us
        self.ratio = ratio

    def forward(self, input):
        in_channels = self.in_channels_basic   # 就是普通的 in_channels
        out_channels = self.out_channels_basic  # 就是普通的 out_channels
        #   这两个判断是根据width_mult缩小输入输出通道
        if self.us[0]:
            in_channels = int(make_divisible(
                self.in_channels_basic
                * self.width_mult
                / self.ratio[0]) * self.ratio[0])
        if self.us[1]:
            out_channels = int(make_divisible(
                self.out_channels_basic
                * self.width_mult
                / self.ratio[1]) * self.ratio[1])
        self.groups = in_channels if self.depthwise else 1
        #   也就是父类的权重呗(nn.Conv2d)(但是感觉多此一举啊，因为nn.Conv2d初始化的时候已经是缩小的了)# 只截取部分的卷积权重*****
        #   经过实验确实多此一举，这里形状是一样的
        #   我猜测这样只不过是为了把全中提取出来，不用这个自带卷积，而是用下面的nn.functional.conv2d（猜测）
        weight = self.weight[:out_channels, :in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        if getattr(FLAGS, 'conv_averaged', False):  #   没有conv_averaged这个参数
            y = y * (max(self.in_channels_list)/self.in_channels)   #这个貌似是类似于归一化的东西把
        return y


class USLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, us=[True, True],previous_channel=256):
        in_features_max = in_features
        out_features_max = out_features
        if us[0]:   #   width_mult是全局外定义的，确实是最大的，这个判断是为了把输入输出变成能被8整除的
            in_features_max = make_divisible(
                in_features * width_mult)
        if us[1]:
            out_features_max = make_divisible(
                out_features * width_mult)
        #   初始化父类线性层（nn.Linear）
        #   USLinear初始化也就比nn.Linear多了几个参数而已。然后重写forward函数
        super(USLinear, self).__init__(
            in_features_max, out_features_max, bias=bias)
        self.in_features_basic = in_features
        self.out_features_basic = out_features
        self.width_mult = None
        self.us = us
        self.previous_channel=previous_channel

    def forward(self, input):
        in_features = self.in_features_basic
        out_features = self.out_features_basic
        if self.us[0]:      #   这和上面初始化的两个判断本质上是一直的
            in_features = make_divisible(
                self.previous_channel * self.width_mult)*int(self.in_features_basic/self.previous_channel)   #   这不能自己加，加了的话别的分辨率就不对了
        if self.us[1]:
            out_features = make_divisible(
                self.out_features_basic * self.width_mult)
        weight = self.weight[:out_features, :in_features]
        if self.bias is not None:
            bias = self.bias[:out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)


class USBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, ratio=1):
        num_features_max = int(make_divisible(      #   width_mult是全局外定义的，确实是最大的，这个判断是为了把输入输出变成能被8整除的
            num_features * width_mult / ratio) * ratio)
        super(USBatchNorm2d, self).__init__(
            num_features_max, affine=True, track_running_stats=False)
        self.num_features_basic = num_features
        # for tracking log during training
        self.bn = nn.ModuleList(    #这句话意味着每一个BN层都是有所有宽度的BN层参数
            [nn.BatchNorm2d(i, affine=False)
             for i in [
                     int(make_divisible(
                         num_features * width_mult / ratio) * ratio)
                     for width_mult in FLAGS.width_mult_list]
             ]
        )
        self.ratio = ratio
        self.width_mult = None
        self.ignore_model_profiling = True

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        c = int(make_divisible(
            self.num_features_basic * self.width_mult / self.ratio) * self.ratio)
        if self.width_mult in FLAGS.width_mult_list:
            idx = FLAGS.width_mult_list.index(self.width_mult)
            y = nn.functional.batch_norm(
                input,
                self.bn[idx].running_mean[:c],
                self.bn[idx].running_var[:c],   # mean var截取一部分
                weight[:c], # weight，bias截取一部分
                bias[:c],
                self.training,
                self.momentum,
                self.eps)
        else:   #   这是什么情况？？
            y = nn.functional.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                weight[:c],
                bias[:c],
                self.training,
                self.momentum,
                self.eps)
        return y

