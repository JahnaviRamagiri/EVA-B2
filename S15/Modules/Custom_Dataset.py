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

class CustomDataSet(Dataset):
    def __init__(self, start_no=1, end_no=101):
        self.fg_bg = []
        self.bg = []
        self.fg_bg_depth = []
        self.fg_bg_masks = []

        for bg_number in range(start_no, end_no):
            print("start")
            index_no = (bg_number - 1) * 4000 + 1
            for count in range(0, 4000):
                self.bg.append(f'/content/drive/My Drive/Utils/background/{str(bg_number)}.jpg')
                self.fg_bg.append(f'/content/drive/My Drive/Utils/fg_bg/fg_bg{str(bg_number)}/fg_bg{str(index_no)}.jpg')
                self.fg_bg_depth.append(
                    f'/content/drive/My Drive/Utils/fg_bg_depth/fg_bg_depth{str(bg_number)}/depth_fg_bg{str(index_no)}.jpg')
                self.fg_bg_masks.append(
                    f'/content/drive/My Drive/Utils/fg_bg_mask/fg_bg_mask{str(bg_number)}/fg_bg_masks{str(index_no)}.jpg')
                index_no += 1
        print("end")

    def __len__(self):
        return len(self.fg_bg)

    def __getitem__(self, idx):
        fg_bg = self.fg_bg[idx]
        fg_bg_masks = self.fg_bg_masks[idx]
        fg_bg_depth = self.fg_bg_depth[idx]
        bg = self.bg[idx]
        return fg_bg, bg, fg_bg_masks, fg_bg_depth


"""### Custom Train Dataset"""


class CustomTrainDataSet:
    def __init__(self, custom_data_set=None, transform=None):
        self.custom_data_set = custom_data_set
        self.transform = transform

    def __getitem__(self, index):
        fg_bg, bg, fg_bg_masks, fg_bg_depth = self.custom_data_set[index]

        fg_bg = Image.open(f'{fg_bg}')
        bg = Image.open(f'{bg}')
        fg_bg_depth = np.asarray(Image.open(f'{fg_bg_depth}').convert('L'))
        fg_bg_masks = np.asarray(Image.open(f'{fg_bg_masks}').convert('L'))
        # fg_bg_depth = fg_bg_depth[:,:,:]
        # fg_bg_masks = fg_bg_masks[:,:,:]

        input_img = np.concatenate((bg, fg_bg), axis=2)

        if self.transform:
            input_img = self.transform['input'](input_img)
            fg_bg_depth = self.transform['depth'](fg_bg_depth)
            fg_bg_masks = self.transform['mask'](fg_bg_masks)

        return input_img, fg_bg_masks, fg_bg_depth

    def __len__(self):
        return len(self.custom_data_set)

def get_train_test_data(split = 70, train_transforms = None, test_transforms = None, start_no=1, end_no = 101):
    custom_data_set = CustomDataSet(start_no, end_no)
    train_len = len(custom_data_set) * split//100
    test_len = len(custom_data_set) - train_len
    train_set, val_set = random_split(custom_data_set, [train_len, test_len])
    train_dataset = CustomTrainDataSet(train_set, transform=train_transforms)
    test_dataset = CustomTrainDataSet(val_set, transform=test_transforms)
    return train_dataset, test_dataset
