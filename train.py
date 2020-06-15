##########################
# Nicola Altini (2020)
# V-Net for Hippocampus Segmentation from MRI with PyTorch
##########################

##########################
# Imports
##########################
import os
import numpy as np
import pickle
import torch
import torch.optim as optim
from sklearn.model_selection import KFold

##########################
# Local Imports
##########################
from config import *
from semseg.utils import train_model, val_model
from semseg.data_loader import GetDataLoader3DTraining, GenDataLoader3DValidation
from models.vnet3d import VNet3D

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
train_data_loader_3D = GetDataLoader3DTraining(config)
iterable_data_loader = iter(train_data_loader_3D)
inputs,labels = next(iterable_data_loader)
print("Shape of Batch: [input {}] [label {}]".format(inputs.shape, labels.shape))

##########################
# Check Net
##########################
net = VNet3D(num_outs=config.num_outs, channels=8)
outputs = net(inputs)
print("Shape of Output: [output {}]".format(outputs.shape))

##########################
# Training loop
##########################
cuda_dev = torch.device('cuda')
optimizer = optim.Adam(net.parameters(), lr=config.lr)

net = net.to(cuda_dev)

if config.do_crossval:
    ##########################
    # Training (cross-validation)
    ##########################
    num_val_images = num_train_images // config.num_folders

    multi_dices_crossval = list()
    mean_multi_dice_crossval = list()
    std_multi_dice_crossval = list()

    kf = KFold(n_splits=config.num_folders)
    for idx, (train_index, val_index) in enumerate(kf.split(test_images)):
        print("[Folder {:d}]".format(idx))
        print("TRAIN:", train_index, "VAL:", val_index)
        train_images_np,train_labels_np = np.array(train_images), np.array(train_labels)
        train_images_list = list(train_images_np[train_index])
        train_images_list = [os.path.join(train_images_folder, train_image) for train_image in train_images_list]
        val_images_list = list(train_images_np[val_index])
        val_images_list = [os.path.join(train_images_folder, val_image) for val_image in val_images_list]
        train_labels_list = list(train_labels_np[train_index])
        train_labels_list = [os.path.join(train_labels_folder, train_label) for train_label in train_labels_list]
        val_labels_list = list(train_labels_np[val_index])
        val_labels_list = [os.path.join(train_labels_folder, val_label) for val_label in val_labels_list]
        config.train_images, config.val_images = train_images_list, val_images_list
        config.train_labels, config.val_labels = train_labels_list, val_labels_list

        ##########################
        # Training (cross-validation)
        ##########################
        net = VNet3D(num_outs=config.num_outs, channels=8)
        train_data_loader_3D = GetDataLoader3DTraining(config)
        net = train_model(net, optimizer, train_data_loader_3D,
                          config, device=cuda_dev, logs_folder=logs_folder)

        ##########################
        # Validation (cross-validation)
        ##########################
        val_data_loader_3D = GenDataLoader3DValidation(config)
        multi_dices, mean_multi_dice, std_multi_dice = val_model(net, val_data_loader_3D,
                                                                 config, device=cuda_dev)
        multi_dices_crossval.append(multi_dices)
        mean_multi_dice_crossval.append(mean_multi_dice)
        std_multi_dice_crossval.append(std_multi_dice)
        torch.save(net, os.path.join(logs_folder, "model_folder_{:d}.pt".format(idx)))


##########################
# Training (full training set)
##########################
config.train_images = [os.path.join(train_images_folder, train_image)
                       for train_image in train_images]
config.train_labels = [os.path.join(train_labels_folder, train_label)
                       for train_label in train_labels]
train_data_loader_3D = GetDataLoader3DTraining(config)
net = train_model(net, optimizer, train_data_loader_3D,
                  config, device=cuda_dev, logs_folder=logs_folder)

torch.save(net,os.path.join(logs_folder,"model.pt"))


