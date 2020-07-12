import copy
import os

import numpy as np

from semseg.data_loader import TorchIODataLoader3DTraining
from models.vnet3d import VNet3D


def print_config(config):
    attributes_config = [attr for attr in dir(config)
                         if not attr.startswith('__')]
    print("Train Config")
    for item in attributes_config:
        print("{:15s} ==> {}".format(item, getattr(config, item)))


def check_train_set(config):
    num_train_images = len(config.train_images)
    num_train_labels = len(config.train_labels)

    assert num_train_images == num_train_labels, "Mismatch in number of training images and labels!"

    print("There are: {} Training Images".format(num_train_images))
    print("There are: {} Training Labels".format(num_train_labels))


def check_torch_loader(config, check_net=False):
    train_data_loader_3D = TorchIODataLoader3DTraining(config)
    iterable_data_loader = iter(train_data_loader_3D)
    el = next(iterable_data_loader)
    inputs, labels = el['t1']['data'], el['label']['data']
    print("Shape of Batch: [input {}] [label {}]".format(inputs.shape, labels.shape))
    if check_net:
        net = VNet3D(num_outs=config.num_outs, channels=config.num_channels)
        outputs = net(inputs)
        print("Shape of Output: [output {}]".format(outputs.shape))


def print_folder(idx, train_index, val_index):
    print("+==================+")
    print("+ Cross Validation +")
    print("+     Folder {:d}     +".format(idx))
    print("+==================+")
    print("TRAIN [Images: {:3d}]:\n{}".format(len(train_index), train_index))
    print("VAL   [Images: {:3d}]:\n{}".format(len(val_index), val_index))


def train_val_split(train_images, train_labels, train_index, val_index):
    train_images_np, train_labels_np = np.array(train_images), np.array(train_labels)
    train_images_list = list(train_images_np[train_index])
    val_images_list = list(train_images_np[val_index])
    train_labels_list = list(train_labels_np[train_index])
    val_labels_list = list(train_labels_np[val_index])
    return train_images_list, val_images_list, train_labels_list, val_labels_list


def train_val_split_config(config, train_index, val_index):
    train_images_list, val_images_list, train_labels_list, val_labels_list = \
        train_val_split(config.train_images, config.train_labels, train_index, val_index)
    new_config = copy.copy(config)
    new_config.train_images, new_config.val_images = train_images_list, val_images_list
    new_config.train_labels, new_config.val_labels = train_labels_list, val_labels_list
    return new_config
