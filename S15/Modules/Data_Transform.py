from torch.utils.data import Dataset, random_split
import math
from PIL import Image
import cv2
import numpy as np
import torch
import os
from tqdm import notebook
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import albumentations as A
import albumentations.pytorch as AP
import random
from tqdm.notebook import tqdm
from tqdm import tqdm_notebook
import time
import matplotlib.pyplot as plt

class Albumentation:
    """
    Helper class to create test and train transforms using Albumentations
    """

    def __init__(self, transforms=[]):
        self.transforms = A.Compose(transforms)

    def __call__(self, img):
        img = np.array(img)
        return self.transforms(image=img)['image']

def get_train_test_data(split=70, train_transforms=None, test_transforms=None, start_no=1, end_no=101):
    custom_data_set = CustomDataSet(start_no, end_no)
    train_len = len(custom_data_set) * split // 100
    test_len = len(custom_data_set) - train_len
    train_set, val_set = random_split(custom_data_set, [train_len, test_len])
    train_dataset = CustomTrainDataSet(train_set, transform=train_transforms)
    test_dataset = CustomTrainDataSet(val_set, transform=test_transforms)
    return train_dataset, test_dataset


train_dataset, test_dataset = get_train_test_data(start_no=20, end_no=21, train_transforms=transforms,
                                                  test_transforms=transforms)
print(len(train_dataset))
print(len(test_dataset))

"""### Get Data Loader"""


def get_data_loader(train_set, test_set, seed=1, batch_size=8, num_workers=4, pin_memory=True):
    SEED = 1
    cuda = torch.cuda.is_available()
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)
    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers,
                           pin_memory=pin_memory) if cuda else dict(shuffle=True, batch_size=64)
    train_loader = torch.utils.data.DataLoader(train_set, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_set, **dataloader_args)
    return train_loader, test_loader


train_loader, test_loader = get_data_loader(train_dataset, test_dataset)
