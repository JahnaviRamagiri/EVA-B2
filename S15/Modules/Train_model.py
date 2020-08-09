import torch
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


def train_model(model, device, trainloader, testloader, optimizer, mask_criterion, depth_criterion, EPOCHS,
                scheduler=False, batch_scheduler=False, best_loss=1000,
                path="/content/gdrive/My Drive/API/bestmodel.pt"):
    start = time.time()
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch + 1, 'LR:', optimizer.param_groups[0]['lr'])
        LR.append(optimizer.param_groups[0]['lr'])
        train_scheduler = False

        if (batch_scheduler):
            train_scheduler = scheduler
        train_loss, train_acc = train(model, device, trainloader, optimizer, mask_criterion, depth_criterion, epoch,
                                      train_scheduler)

        test_loss, test_acc = test(model, device, mask_criterion, depth_criterion, testloader, epoch)
        if (scheduler and not batch_scheduler and not isinstance(scheduler,
                                                                 torch.optim.lr_scheduler.ReduceLROnPlateau)):
            scheduler.step()

        elif (scheduler and not batch_scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
            scheduler.step(test_loss[-1][2])

        # if(test_loss[-1][2]<best_loss):
        #   print("loss reduced, Saving model....")
        #   best_loss = test_loss[-1][2]
        #   torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': test_loss,
        #         'acc' : test_acc
        #         }, path,pickle_module=dill)

        print('*' * 30)

    end = time.time()
    print(f"Training time: {round((end - start), 3)}s {EPOCHS} epochs")