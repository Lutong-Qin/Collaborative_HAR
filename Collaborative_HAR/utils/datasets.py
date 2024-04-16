import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils.config import FLAGS
from torchvision import datasets, transforms
from utils.transforms import InputList_3
from collections import Counter
import torch.utils.data as Data



def get_dataset():
    if 'uci' in FLAGS.dataset:
        return get_har('uci')
    if 'wisdm' in FLAGS.dataset:
        return get_har('wisdm')
    if 'oppo' in FLAGS.dataset:
        return get_har('oppo')
    if 'unimib' in FLAGS.dataset:
        return get_har('unimib')
    if 'pamap2' in FLAGS.dataset:
        return get_har('pamap2')
    if 'usc' in FLAGS.dataset:
        return get_har('usc')
    else:
        raise NotImplementedError('dataset not implemented.')


class HAR(Dataset):
    def __init__(self, root_dir, train_or_test, transform=None):
        # self 指定了一个类当中的全局变量，该变量可以让后面的函数使用
        self.root_dir = root_dir
        self.label_dir = []
        # self.label_dir = data_label_dir#是个列表如：['x_test.npy','y_test.npy']
        self.label_dir.append(f'x_{train_or_test}.npy')
        self.label_dir.append(f'y_{train_or_test}.npy')
        self.transform = transform
        self.path=[]
        for i in range(len(self.label_dir)):
            self.path.append(os.path.join(self.root_dir, self.label_dir[i]))
        self.x = torch.from_numpy(np.load(self.path[0])).float()
        self.y = torch.from_numpy(np.load(self.path[1])).long()
        self.x = torch.unsqueeze(self.x, 1)

        self.data = Data.TensorDataset(self.x, self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.y)

def get_har(dataset='uci'):
    train_dataset=HAR(f'/data1/experiment/qinlutong666/sftp/Datasets/{dataset}','train',InputList_3(FLAGS.resolution_list))
    test_dataset = HAR(f'/data1/experiment/qinlutong666/sftp/Datasets/{dataset}','test')
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=FLAGS.batch_size, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=FLAGS.batch_size, shuffle=True)
    return train_loader, test_loader


