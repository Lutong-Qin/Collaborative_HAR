import argparse
import importlib
import os
import time
import random

import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from numpy import mean

import ComputePostBN
from utils.setlogger import get_logger, get_logger_result

from utils.model_profiling import model_profiling
from utils.config import FLAGS
from utils.datasets import get_dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--save', default='save/unimib/default-{}'.format(time.time()),
                    type=str, metavar='SAVE',
                    help='path to the experiment logging directory'
                         '(default: save/debug)')
args = parser.parse_args()


def set_random_seed():
    """set random seed"""
    seed = kk
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_model():
    """get model"""
    model_lib = importlib.import_module(FLAGS.model)
    if "resnet" in FLAGS.model:
        model = model_lib.ResNet(1, FLAGS.num_classes)
    else:
        model = model_lib.CNN(1, FLAGS.num_classes)
    return model


def get_optimizer(model):
    """get optimizer"""
    if FLAGS.dataset == 'uci' or FLAGS.dataset == 'wisdm' \
            or FLAGS.dataset == 'unimib' or FLAGS.dataset == 'pamap2' \
            or FLAGS.dataset == 'oppo30' or FLAGS.dataset == 'usc':
        optimizer = torch.optim.Adam(model.parameters(), float(FLAGS.lr))

    else:
        optimizer = torch.optim.SGD(model.parameters(), FLAGS.lr,
                                    momentum=FLAGS.momentum, nesterov=FLAGS.nesterov,
                                    weight_decay=FLAGS.weight_decay)
    return optimizer


def profiling(model, use_cuda):
    """profiling on either gpu or cpu"""

    for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
        for resolution in FLAGS.resolution_list:
            model.apply(
                lambda m: setattr(m, 'width_mult', width_mult))
            n_macs, n_params = model_profiling(model, resolution[0], resolution[1], verbose=True)
            flops = n_macs / 1e6
            params = n_params / 1e6
            model_size.append({'width_mult': width_mult, 'resolution': resolution, 'flops': flops, 'params': params})
    new_sys2 = sorted(model_size.copy(), key=lambda e: (e.__getitem__('width_mult'), e.__getitem__('flops')),
                      reverse=True)
    for m in new_sys2:
        temporary = list(m.values())
        logger_result.info(f'{temporary[0]}-{temporary[1]}-params:{temporary[3]}M-flops:{temporary[2]}M')


def train(epoch, loader, model, criterion, optimizer, lr_scheduler):
    T = 1
    alpha = 0.5
    t_start = time.time()
    model.train()
    for batch_idx, (input_list, target) in enumerate(loader):
        target = target.cuda(non_blocking=True)
        optimizer.zero_grad()
        # do max width
        max_width = FLAGS.width_mult_range[1]
        model.apply(lambda m: setattr(m, 'width_mult', max_width))
        max_output = model(input_list[0])
        loss = criterion(max_output, target)
        loss.backward()
        max_output_detach = max_output.detach()
        min_width = FLAGS.width_mult_range[0]
        width_mult_list = [min_width, max_width]
        sampled_width = list(
            np.random.uniform(FLAGS.width_mult_range[0], FLAGS.width_mult_range[1], 6))
        width_mult_list.extend(sampled_width)
        lenght_resolution = len(FLAGS.resolution_list)
        for width_mult in sorted(width_mult_list, reverse=True):
            model.apply(
                lambda m: setattr(m, 'width_mult', width_mult))
            output = model(input_list[random.randint(0, lenght_resolution - 1)])
            loss = torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output / T, dim=1),
                                                             F.softmax(max_output_detach / T, dim=1)) * (alpha * T * T) \
                   + (1. - alpha) * criterion(output, target)
            loss.backward()
        optimizer.step()
        if batch_idx % FLAGS.print_freq == -1 or batch_idx == len(loader) - 1:
            with torch.no_grad():
                for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
                    model.apply(lambda m: setattr(m, 'width_mult', width_mult))
                    output = model(input_list[0])
                    loss = criterion(output, target).cpu().numpy()
                    indices = torch.max(output, dim=1)[1]
                    acc = (indices == target).sum().cpu().numpy() / indices.size()[0]
                    logger.info('TRAIN {:.1f}s LR:{:.4f} {}x Epoch:{}/{} Iter:{}/{} Loss:{:.4f} Acc:{:.3f}'.format(
                        time.time() - t_start, optimizer.param_groups[0]['lr'], str(width_mult), epoch,
                        FLAGS.num_epochs, batch_idx, len(loader), loss, acc))


def validate(epoch, loader, model, criterion,
             postloader):  # postloader就是训练集   loader是val_loader 验证只要祟拜你验证一下就行了，太多容易减慢速度
    t_start = time.time()
    model.eval()
    resolution = FLAGS.image_size  # 32
    with torch.no_grad():
        for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
            model.apply(lambda m: setattr(m, 'width_mult', width_mult))
            model = ComputePostBN.ComputeBN(model, postloader, resolution)
            loss, acc, cnt = 0, 0, 0
            for batch_idx, (input, target) in enumerate(loader):
                input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)
                output = model(input)
                loss += criterion(output, target).cpu().numpy() * target.size()[0]
                indices = torch.max(output, dim=1)[1]
                acc += (indices == target).sum().cpu().numpy()
                cnt += target.size()[0]
            logger.info('VAL {:.1f}s {}x Epoch:{}/{} Loss:{:.4f} Acc:{:.3f}'.format(
                time.time() - t_start, str(width_mult), epoch,
                FLAGS.num_epochs, loss / cnt, acc / cnt * 100))


def test(epoch, loader, model, criterion, postloader):
    t_start = time.time()
    model.eval()
    lenght_resolution = len(FLAGS.resolution_list)
    with torch.no_grad():
        for i, width_mult in enumerate(sorted(FLAGS.width_mult_list, reverse=True)):
            for j, resolution in enumerate(FLAGS.resolution_list):

                model.apply(lambda m: setattr(m, 'width_mult', width_mult))
                model = ComputePostBN.ComputeBN(model, postloader, resolution)

                flops = model_size[i * (lenght_resolution) + j]['flops']

                loss, acc, cnt = 0, 0, 0
                for batch_idx, (input, target) in enumerate(loader):
                    input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)
                    output = model(
                        F.interpolate(input, (resolution[0], resolution[1]), mode='nearest-exact'))
                    loss += criterion(output, target).cpu().numpy() * target.size()[0]
                    indices = torch.max(output, dim=1)[1]
                    acc += (indices == target).sum().cpu().numpy()
                    cnt += target.size()[0]
                loss = loss / cnt
                acc = acc / cnt * 100

                result_test_loss[i][j].append(loss)
                if epoch == FLAGS.num_epochs-1:
                    result_test_acc[i][j].append(acc)

                logger.info('TEST {:.1f}s {}x-{}-{} Epoch:{}/{} Loss:{:.4f} Acc:{:.3f}'.format(
                    time.time() - t_start, str(width_mult), str(resolution), str(flops), epoch,
                    FLAGS.num_epochs, loss, acc))


def train_val_test():
    """train and val"""
    model = get_model()
    model_wrapper = torch.nn.DataParallel(model).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    train_loader, val_loader = get_dataset()
    if FLAGS.pretrained:
        checkpoint = torch.load(FLAGS.pretrained)
        if type(checkpoint) == dict and 'model' in checkpoint:
            checkpoint = checkpoint['model']
        new_keys = list(model_wrapper.state_dict().keys())
        old_keys = list(checkpoint.keys())
        new_keys = [key for key in new_keys if 'running' not in key]
        new_keys = [key for key in new_keys if 'tracked' not in key]
        old_keys = [key for key in old_keys if 'running' not in key]
        old_keys = [key for key in old_keys if 'tracked' not in key]
        if not FLAGS.test_only:
            old_keys = old_keys[:-2]
            new_keys = new_keys[:-2]

        new_checkpoint = {}
        for key_new, key_old in zip(new_keys, old_keys):
            new_checkpoint[key_new] = checkpoint[key_old]
        model_wrapper.load_state_dict(new_checkpoint, strict=False)
        print('Loaded model {}.'.format(FLAGS.pretrained))
    optimizer = get_optimizer(model_wrapper)

    if FLAGS.resume:
        checkpoint = torch.load(FLAGS.resume)
        model_wrapper.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['last_epoch']
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * FLAGS.num_epochs)
        lr_scheduler.last_epoch = last_epoch
        print('Loaded checkpoint {} at epoch {}.'.format(
            FLAGS.resume, last_epoch))
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                  len(train_loader) * FLAGS.num_epochs)  # 这个并没有用到
        last_epoch = lr_scheduler.last_epoch
        print(model_wrapper)
        if FLAGS.profiling:
            if 'gpu' in FLAGS.profiling:
                profiling(model, use_cuda=True)
            if 'cpu' in FLAGS.profiling:
                profiling(model, use_cuda=False)

    if FLAGS.test_only:
        logger.info('Start testing.')
        test(last_epoch, val_loader, model_wrapper, criterion, train_loader)
        return

    logger.info('Start training.')
    for epoch in range(last_epoch + 1, FLAGS.num_epochs):

        adjust_learning_rate(optimizer, epoch)
        train(epoch, train_loader, model_wrapper, criterion, optimizer, lr_scheduler)  # 训练

        validate(epoch, val_loader, model_wrapper, criterion, train_loader)


        # torch.save(
        #     {
        #         'model': model_wrapper.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'last_epoch': epoch,
        #     },
        #     os.path.join(saved_path, 'checkpoint_{}.pt'.format(epoch)))
    test(FLAGS.num_epochs-1, val_loader, model_wrapper, criterion, train_loader)
    lenght_resolution = len(FLAGS.resolution_list)
    with open(os.path.join(args.save, 'result.txt'), 'w') as fout:
        with open(os.path.join(args.save, 'flops_acc_test_is_right.txt'), 'w') as f_a:
            for i, width_mult in enumerate(sorted(FLAGS.width_mult_list, reverse=True)):
                for j, resolution in enumerate(FLAGS.resolution_list):
                    avg_result = mean(result_test_acc[i][j])
                    flops = model_size[i * lenght_resolution + j]['flops']
                    width = model_size[i * lenght_resolution + j]['width_mult']
                    resolu = model_size[i * lenght_resolution + j]['resolution']
                    params = model_size[i * lenght_resolution + j]['params']
                    assert width_mult == width
                    assert resolution == resolu
                    flops = round(flops, 2)
                    params = round(params, 2)
                    logger_result.info(f'{width_mult}-{resolution}-params:{params}M-flops:{flops}M  acc:{avg_result}')

                    # last_draw_result_acc.append(avg_result)
                    # last_draw_result_flops.append(flops)

                    avg_result = round(avg_result, 2)
                    fout.write(f'{width_mult}\t{resolution}\t{flops}\t{avg_result}\n')
                    f_a.write(f'{flops}\t{params}\t{avg_result}\n')

    anytime_dir = os.path.join(args.save, 'result.txt')
    f = np.loadtxt(anytime_dir, delimiter='\t', usecols=(2, 3))
    last_draw_result_flops = f[:, 0]
    last_draw_result_acc = f[:, 1]

    index = np.array(last_draw_result_flops).argsort()
    last_draw_result_flops = sorted(last_draw_result_flops, reverse=False)
    acc_copy = last_draw_result_acc.copy()
    for i, j in enumerate(index):
        last_draw_result_acc[i] = acc_copy[j]

    with open(os.path.join(args.save, 'flops_acc.txt'), 'a') as f_a_t:
        for i, j in enumerate(last_draw_result_flops):
            f_a_t.write(f'{last_draw_result_flops[i]}\t{last_draw_result_acc[i]}\n')

    return


def adjust_learning_rate(optimizer, epoch):
    lr = float(FLAGS.lr) * (0.1 ** (epoch // 50))
    optimizer.param_groups[0]['lr'] = lr


def main():
    """train and eval model"""
    set_random_seed()
    train_val_test()


if __name__ == "__main__":

    for kk in [2]:
        args.save = args.save[:-1] + f'{kk}'
        if not os.path.exists(args.save):
            os.makedirs(args.save)

        # set log files
        saved_path = os.path.join(args.save, "logs", '{}-{}'.format(FLAGS.dataset, FLAGS.model[7:]))
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)
        logger = get_logger(
            os.path.join(saved_path, '{}_div1optimizer.log'.format('test' if FLAGS.test_only else 'train')))

        if not os.path.exists(args.save):
            os.makedirs(args.save)
        logger_result = get_logger_result(os.path.join(args.save, f'result.log'), 'result')
        # **************

        # *********数据保存列表
        result_test_acc = []
        result_test_loss = []
        result_train_acc = []
        result_train_loss = []
        for i, width_mult in enumerate(sorted(FLAGS.width_mult_list, reverse=True)):
            result_test_acc.append([])
            result_test_loss.append([])
            result_train_acc.append([])
            result_train_loss.append([])
            for j, resolution in enumerate(FLAGS.resolution_list):
                result_test_acc[i].append([])
                result_test_loss[i].append([])
                result_train_acc[i].append([])
                result_train_loss[i].append([])
        # *********

        model_size = []

        main()
