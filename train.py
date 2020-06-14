##########################
# Nicola Altini (2020)
# V-Net for Hippocampus Segmentation from MRI with PyTorch
##########################

##########################
# Imports
##########################
import os
import numpy as np
import torch
import torch.optim as optim

##########################
# Local Imports
##########################
from config import *
from semseg.utils import GetDataLoader3D, train_model
from models.vnet3d import VXNet3D

##########################
# Check training set
##########################
num_train_images = len(train_images)
num_train_labels = len(train_labels)

assert num_train_images == num_train_labels, "Mismatch in number of training images and labels!"

print("There are: {} Training Images".format(num_train_images))
print("There are: {} Training Labels".format(num_train_labels))


##########################
# Config
##########################
config = SemSegMRIConfig()
attributes_config = [attr for attr in dir(config)
                     if not attr.startswith('__')]
print("Train Config")
for item in attributes_config:
    print("{:15s} ==> {}".format(item, getattr(config, item)))

##########################
# Check Torch Dataset and DataLoader
##########################
train_data_loader_3D = GetDataLoader3D(config)
iterable_data_loader = iter(train_data_loader_3D)
inputs,labels = next(iterable_data_loader)
print("Shape of Batch: [input {}] [label {}]".format(inputs.shape, labels.shape))

net = VXNet3D(num_outs=3)
outputs = net(inputs)
print("Shape of Output: [output {}]".format(outputs.shape))

##########################
# Training loop
##########################
cuda_dev = torch.device('cuda')
optimizer = optim.Adam(net.parameters(), lr=config.lr)

net = net.to(cuda_dev)

net = train_model(net, optimizer, train_data_loader_3D,
                  config, device=cuda_dev, logs_folder=logs_folder)
