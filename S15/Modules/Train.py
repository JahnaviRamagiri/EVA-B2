import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
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

train_losses = []
test_losses = []
train_acc = []
test_acc = []


def train(model, device, train_loader, optimizer, mask_criterion, depth_criterion, epoch, scheduler=False):
    running_mask_loss = 0
    running_depth_loss = 0
    model.train()
    pbar = tqdm(train_loader)
    total_length = len(train_loader)
    print("total length", total_length)
    for batch_idx, (data, mask_target, depth_target) in enumerate(pbar):
        # get samples
        data, mask_target, depth_target = data.to(device), mask_target.to(device), depth_target.to(device)

        optimizer.zero_grad()

        mask_target = mask_target.unsqueeze_(1)
        depth_target = depth_target.unsqueeze_(1)
        mask_pred, depth_pred = model(data)
        mask_target = torch.sigmoid(mask_target)
        depth_target = torch.sigmoid(depth_target)
        # Calculate loss

        mask_loss = mask_criterion(mask_pred, mask_target, )
        depth_loss = depth_criterion(depth_pred, depth_target)
        # mask_loss = mask_loss.item()
        # depth_loss = depth_loss.item()
        loss = mask_loss + depth_loss
        running_mask_loss += mask_loss.item()
        running_depth_loss += depth_loss.item()

        # Backpropagation
        torch.autograd.backward([mask_loss, depth_loss])

        optimizer.step()
        if (scheduler):
            scheduler.step()

        pbar.set_description(f'Loss={loss:0.4f}')
    # print('\nTrain set: Average loss: {:.4f}'.format(str(mask_loss+depth_loss)/2))
    average_loss = (running_mask_loss + running_depth_loss) / 2 * total_length
    print('average_loss', average_loss)
    # train_losses.append((mask_loss/total_length,depth_loss/total_length))
    train_losses.append((running_mask_loss / total_length, running_depth_loss / total_length))

    print("train loss")
    print(train_losses)
    return train_losses, train_acc
