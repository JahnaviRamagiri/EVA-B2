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


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


images_list = []
mask_target_list = []
depth_target_list = []
mask_pred_list = []
depth_pred_list = []


def show_result_img(target, predict, type, name):
    fig, a = plt.subplots(2, 1, figsize=(45, 35))
    fig.suptitle(type + " " + name, fontweight="bold", fontsize=45, y=1.1, color='r')

    target = target * 255
    target = target.numpy()

    predict = predict * 255
    predict = predict.numpy()

    plt.axis("off")
    a[0].imshow(target[0], cmap="gray")
    a[1].imshow(predict[0], cmap="gray")

    a[0].set_title('Target ' + type, fontsize=35)
    a[1].set_title('Predicted ' + type, fontsize=35)
    a[0].axis("off")
    a[1].axis("off")


def show_results(model, testloader, device, name):
    batch = next(iter(testloader))
    images, mask_target, depth_target = batch

    mask, depth = model(images.to(device))
    batch_preds_mask = mask  # torch.sigmoid(mask)
    batch_preds_mask = batch_preds_mask.detach().cpu()

    batch_preds_depth = depth
    batch_preds_depth = batch_preds_depth.detach().cpu()
    plt.axis("off")
    m = torch.unsqueeze(mask_target, 1)
    d = torch.unsqueeze(depth_target, 1)
    images = []
    mask_target = []
    depth_target = []
    mask_pred = []
    depth_pred = []
    for i in range(8):
        mask_target.append(m[i])
        depth_target.append(d[i])
        mask_pred.append(batch_preds_mask[i])
        depth_pred.append(batch_preds_depth[i])
    mask_t = torchvision.utils.make_grid(mask_target, nrow=5, padding=1, scale_each=True)
    mask_p = torchvision.utils.make_grid(mask_pred, nrow=5, padding=1, scale_each=True)
    depth_t = torchvision.utils.make_grid(depth_target, nrow=5, padding=1, scale_each=True)
    depth_p = torchvision.utils.make_grid(depth_pred, nrow=5, padding=1, scale_each=True)
    show_result_img(mask_t, mask_p, "mask", name=name)
    show_result_img(depth_t, depth_p, "depth", name=name)
# show_results(model,testloader,name = "Results_2")