##########################
# Nicola Altini (2020)
# V-Net for Hippocampus Segmentation from MRI with PyTorch
##########################

##########################
# Imports
##########################
import numpy as np
import torch
from sklearn.model_selection import KFold

##########################
# Local Imports
##########################
from config.config import *
from config.paths import logs_folder, train_images, train_labels
from run.utils import train_val_split
from semseg.train import val_model
from semseg.data_loader_torchio import TorchIODataLoader3DValidation

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
config.batch_size = 1
attributes_config = [attr for attr in dir(config)
                     if not attr.startswith('__')]
print("Train Config")
for item in attributes_config:
    print("{:15s} ==> {}".format(item, getattr(config, item)))

path_net = "logs/model.pt"
path_nets_crossval = ["logs/model_folder_{:d}.pt".format(idx) for idx in range(config.num_folders)]

##########################
# Val loop
##########################
cuda_dev = torch.device('cuda')

if config.do_crossval:
    ##########################
    # cross-validation
    ##########################
    num_val_images = num_train_images // config.num_folders

    multi_dices_crossval = list()
    mean_multi_dice_crossval = list()
    std_multi_dice_crossval = list()

    kf = KFold(n_splits=config.num_folders)
    for idx, (train_index, val_index) in enumerate(kf.split(train_images)):
        print("+==================+")
        print("+ Cross Validation +")
        print("+     Folder {:d}     +".format(idx))
        print("+==================+")
        print("TRAIN [Images: {:3d}]:\n{}".format(len(train_index), train_index))
        print("VAL   [Images: {:3d}]:\n{}".format(len(val_index), val_index))
        train_images_list, val_images_list, train_labels_list, val_labels_list = \
            train_val_split(train_images, train_labels, train_index, val_index)
        config.train_images, config.val_images = train_images_list, val_images_list
        config.train_labels, config.val_labels = train_labels_list, val_labels_list

        ##########################
        # Training (cross-validation)
        ##########################
        model_path = path_nets_crossval[idx]
        print("Model: {}".format(model_path))
        net = torch.load(model_path)

        ##########################
        # Validation (cross-validation)
        ##########################
        val_data_loader_3D = TorchIODataLoader3DValidation(config)
        multi_dices, mean_multi_dice, std_multi_dice = val_model(net, val_data_loader_3D,
                                                                 config, device=cuda_dev)
        multi_dices_crossval.append(multi_dices)
        mean_multi_dice_crossval.append(mean_multi_dice)
        std_multi_dice_crossval.append(std_multi_dice)
        torch.save(net, os.path.join(logs_folder, "model_folder_{:d}.pt".format(idx)))

    ##########################
    # Saving Validation Results
    ##########################
    multi_dices_crossval_flatten = [item for sublist in multi_dices_crossval for item in sublist]
    mean_multi_dice_crossval_flatten = np.mean(multi_dices_crossval_flatten)
    std_multi_dice_crossval_flatten = np.std(multi_dices_crossval_flatten)
    print("Multi-Dice: {:.4f} +/- {:.4f}".format(mean_multi_dice_crossval_flatten, std_multi_dice_crossval_flatten))
    # Multi-Dice: 0.8668 +/- 0.0337

