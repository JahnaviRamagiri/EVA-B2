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


def test(model, device, mask_criterion, depth_criterion, test_loader, epoch):
    model.eval()
    mask_loss = 0
    depth_loss = 0
    correct = 0
    total_loss = 0
    total_length = len(test_loader)
    with torch.no_grad():
        for data, mask_target, depth_target in tqdm(test_loader):
            data, mask_target, depth_target = data.to(device), mask_target.to(device), depth_target.to(device)
            mask_target = mask_target.unsqueeze_(1)
            depth_target = depth_target.unsqueeze_(1)
            mask_target = torch.sigmoid(mask_target)
            depth_target = torch.sigmoid(depth_target)
            mask_pred, depth_pred = model(data)
            mask_loss += mask_criterion(depth_pred, mask_target, ).item()
            depth_loss += depth_criterion(depth_pred, mask_target, ).item()
            test_loss = mask_loss + depth_loss
            total_loss += test_loss
            # mask_pred = torch.sigmoid(mask_pred)

            # depth_pred = torch.sigmoid(depth_pred)

            # total_loss += dice_coeff(mask_pred, mask_target).item()

            # total_loss += dice_coeff(depth_pred, depth_target).item()

            # mask_target_list.append(mask_target)
            # depth_target_list.append(depth_target)
            # mask_pred_list.append(mask_pred)
            # depth_pred_list.append(depth_pred)

    test_loss /= (2 * total_length)
    total_loss /= (2 * total_length)

    test_losses.append((mask_loss / total_length, depth_loss / total_length, test_loss))

    print('\nTest set: Average loss: {:.4f}'.format(test_loss))
    return test_losses, test_acc
