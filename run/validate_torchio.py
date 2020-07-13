##########################
# Nicola Altini (2020)
# V-Net for Hippocampus Segmentation from MRI with PyTorch
##########################
# python run/validate_torchio.py
# python run/validate_torchio.py --dir=logs/no_augm_torchio
# python run/validate_torchio.py --dir=path/to/logs/dir --verbose=VERBOSE

##########################
# Imports
##########################
import os
import sys
import argparse
import numpy as np
import torch
from sklearn.model_selection import KFold

##########################
# Local Imports
##########################
current_path_abs = os.path.abspath('.')
sys.path.append(current_path_abs)
print('{} appended to sys!'.format(current_path_abs))

from run.utils import (train_val_split_config, print_folder, print_config, check_train_set)
from config.config import *
from config.paths import logs_folder, train_images, train_labels
from semseg.train import val_model
from semseg.data_loader import TorchIODataLoader3DValidation


def run(logs_dir="logs"):
    config = SemSegMRIConfig()

    ##########################
    # Check training set
    ##########################
    check_train_set(config)

    ##########################
    # Config
    ##########################
    config.batch_size = 1
    print_config(config)

    path_nets_crossval = [os.path.join(logs_dir,"model_folder_{:d}.pt".format(idx))
                          for idx in range(config.num_folders)]

    ##########################
    # Val loop
    ##########################
    cuda_dev = torch.device('cuda')

    if config.do_crossval:
        ##########################
        # cross-validation
        ##########################
        multi_dices_crossval = list()
        mean_multi_dice_crossval = list()
        std_multi_dice_crossval = list()

        kf = KFold(n_splits=config.num_folders)
        for idx, (train_index, val_index) in enumerate(kf.split(train_images)):
            print_folder(idx, train_index, val_index)
            config_crossval = train_val_split_config(config, train_index, val_index)

            ##########################
            # Training (cross-validation)
            ##########################
            model_path = path_nets_crossval[idx]
            print("Model: {}".format(model_path))
            net = torch.load(model_path)

            ##########################
            # Validation (cross-validation)
            ##########################
            val_data_loader_3D = TorchIODataLoader3DValidation(config_crossval)
            multi_dices, mean_multi_dice, std_multi_dice = val_model(net, val_data_loader_3D,
                                                                     config_crossval, device=cuda_dev)
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


############################
# MAIN
############################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Validation (With torchio based Data Loader) "
                                                 "for Hippocampus Segmentation")
    parser.add_argument(
        "-V",
        "--verbose",
        default=False, type=bool,
        help="Boolean flag. Set to true for VERBOSE mode; false otherwise."
    )
    parser.add_argument(
        "-D",
        "--dir",
        default="logs", type=str,
        help="Local path to logs dir"
    )
    parser.add_argument(
        "--net",
        default='vnet',
        help="Specify the network to use [unet | vnet] ** FOR FUTURE RELEASES **"
    )

    args = parser.parse_args()
    run(logs_dir=args.dir)
