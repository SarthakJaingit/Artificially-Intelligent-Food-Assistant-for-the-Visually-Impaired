import numpy as np
import torch
import time
import itertools
from engine import engine_SAM

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2, ToTensor

'''
For the bottom imports you must git clone the following repo in order to build the correct files

!git clone https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
%cd Ranger-Deep-Learning-Optimizer
!pip install -e .
%cd ..
'''
from ranger import Ranger

'''
!git clone https://github.com/davda54/sam.git
%cd sam
import sam
%cd ..
'''
import sam

'''
Input:
mode: train or test
Returns: the transforms
'''

def get_transforms(mode):
    if (mode == "train"):
        return A.Compose([
                          A.OneOf([
                          A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2,
                                         val_shift_limit=0.2, p=0.9),
                          A.RandomBrightnessContrast(brightness_limit=0.2,
                                               contrast_limit=0.2, p=0.9)],p=0.9),
                          A.HorizontalFlip(),
                          A.VerticalFlip(),
                          ToTensorV2()
                          ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    elif (mode == "test"):
        return A.Compose([
                          ToTensorV2()
                          ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        raise ValueError("mode is wrong value can either be train or test")

'''
Input:
mode: batch of uneven data
Returns: batch of even data
'''

def collate_fn(batch):
  return tuple([list(a) for a in zip(*batch)])


'''
Input:
net: The object detection model you need to train (This train function works for SSD and FRMN)
epochs: The number of epochs you want to train the model for
train_loader: The generator for the train datatset
test_loader: The generator for the test dataset
lr: The learning rate for optimizer
weight_decay: The weight decay for optimizer
Returns: The trained neural net
'''

def train(net, epochs, train_loader, test_loader, lr, weight_decay):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Check which parameters can calculate gradients.
    params = [p for p in net.parameters() if p.requires_grad]

    base_optimizer = Ranger
    optimizer = sam.SAM(net.parameters(), base_optimizer, lr = lr, weight_decay = weight_decay)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = len(train_loader) * epochs)

    net.to(device)
    print("Device: {}".format(device))
    print("Optimizer: {}".format(optimizer))

    start_time = time.time()

    for epoch in range(epochs):
      engine_SAM.train_one_epoch(net, optimizer, train_loader, device, epoch, print_freq=10, scheduler = lr_scheduler)
      engine_SAM.evaluate(net, test_loader, device=device)

    print("Time for Total Training {:0.2f}".format(time.time() - start_time))

    return net
