##########################
# Nicola Altini (2020)
# V-Net for Hippocampus Segmentation from MRI with PyTorch
##########################
# python run/validate.py
# python run/validate.py --dir=logs/no_augm_torchio
# python run/validate.py --dir=logs/no_augm_torchio --write=0
# python run/validate.py --dir=path/to/logs/dir --write=WRITE --verbose=VERBOSE

##########################
# Imports
##########################
import torch
import numpy as np
import os
from sklearn.model_selection import KFold
import argparse
import sys

##########################
# Local Imports
##########################
current_path_abs = os.path.abspath('.')
sys.path.append(current_path_abs)
print('{} appended to sys!'.format(current_path_abs))

from config.paths import ( train_images_folder, train_labels_folder, train_prediction_folder,
                           train_images, train_labels,
                           test_images_folder, test_images, test_prediction_folder)
from run.utils import (train_val_split, print_folder, nii_load, sitk_load, nii_write, print_config,
                       sitk_write, print_test, np3d_to_torch5d, torch5d_to_np3d, print_metrics)
from config.config import SemSegMRIConfig
from semseg.utils import multi_dice_coeff, plot_confusion_matrix
from sklearn.metrics import confusion_matrix, f1_score


def run(logs_dir="logs", write_out=False, plot_conf=False):
    ##########################
    # Config
    ##########################
    config = SemSegMRIConfig()
    print_config(config)

    ###########################
    # Load Net
    ###########################
    cuda_dev = torch.device("cuda")

    # Load From State Dict
    # path_net = "logs/model_epoch_0080.pht"
    # net = VNet3D(num_outs=config.num_outs, channels=config.num_channels)
    # net.load_state_dict(torch.load(path_net))

    path_net = os.path.join(logs_dir,"model.pt")
    path_nets_crossval = [os.path.join(logs_dir,"model_folder_{:d}.pt".format(idx))
                          for idx in range(config.num_folders)]

    ###########################
    # Eval Loop
    ###########################
    use_nib = True
    pad_ref = (48,64,48)
    multi_dices = list()
    f1_scores = list()

    os.makedirs(train_prediction_folder, exist_ok=True)
    os.makedirs(test_prediction_folder, exist_ok=True)

    train_and_test = [True, False]
    train_and_test_images = [train_images, test_images]
    train_and_test_images_folder = [train_images_folder, test_images_folder]
    train_and_test_prediction_folder = [train_prediction_folder, test_prediction_folder]
    os.makedirs(train_prediction_folder,exist_ok=True)
    os.makedirs(test_prediction_folder,exist_ok=True)

    train_confusion_matrix = np.zeros((config.num_outs, config.num_outs))

    for train_or_test_images, train_or_test_images_folder, train_or_test_prediction_folder, is_training in \
            zip(train_and_test_images, train_and_test_images_folder, train_and_test_prediction_folder, train_and_test):
        print("Images Folder: {}".format(train_or_test_images_folder))
        print("IsTraining: {}".format(is_training))

        kf = KFold(n_splits=config.num_folders)
        for idx_crossval, (train_index, val_index) in enumerate(kf.split(train_images)):
            if is_training:
                print_folder(idx_crossval, train_index, val_index)
                model_path = path_nets_crossval[idx_crossval]
                print("Model: {}".format(model_path))
                net = torch.load(model_path)
                _, train_or_test_images, _, train_labels_crossval = \
                    train_val_split(train_images, train_labels, train_index, val_index)
            else:
                print_test()
                net = torch.load(path_net)
            net = net.cuda(cuda_dev)
            net.eval()

            for idx, train_image in enumerate(train_or_test_images):
                print("Iter {} on {}".format(idx,len(train_or_test_images)))
                print("Image: {}".format(train_image))
                train_image_path = os.path.join(train_or_test_images_folder, train_image)

                if use_nib:
                    train_image_np, affine = nii_load(train_image_path)
                else:
                    train_image_np, meta_sitk = sitk_load(train_image_path)

                with torch.no_grad():
                    inputs = np3d_to_torch5d(train_image_np, pad_ref, cuda_dev)
                    outputs = net(inputs)
                    outputs_np = torch5d_to_np3d(outputs, train_image_np.shape)

                if write_out:
                    filename_out = os.path.join(train_or_test_prediction_folder, train_image)
                    if use_nib:
                        nii_write(outputs_np, affine, filename_out)
                    else:
                        sitk_write(outputs_np, meta_sitk, filename_out)

                if is_training:
                    train_label = train_labels_crossval[idx]
                    train_label_path = os.path.join(train_labels_folder, train_label)
                    if use_nib:
                        train_label_np, _ = nii_load(train_label_path)
                    else:
                        train_label_np, _ = sitk_load(train_label_path)

                    multi_dice = multi_dice_coeff(np.expand_dims(train_label_np,axis=0),
                                                  np.expand_dims(outputs_np,axis=0),
                                                  config.num_outs)
                    print("Multi Class Dice Coeff = {:.4f}".format(multi_dice))
                    multi_dices.append(multi_dice)

                    f1_score_idx = f1_score(train_label_np.flatten(), outputs_np.flatten(), average=None)
                    cm_idx = confusion_matrix(train_label_np.flatten(), outputs_np.flatten())
                    train_confusion_matrix += cm_idx
                    f1_scores.append(f1_score_idx)

            if not is_training:
                break

    print_metrics(multi_dices, f1_scores, train_confusion_matrix)

    if plot_conf:
        plot_confusion_matrix(train_confusion_matrix,
                              target_names=None, title='Cross-Validation Confusion matrix',
                              cmap=None, normalize=False, already_normalized=False,
                              path_out="images/conf_matrix_no_norm_no_augm_torchio.png")
        plot_confusion_matrix(train_confusion_matrix,
                              target_names=None, title='Cross-Validation Confusion matrix (row-normalized)',
                              cmap=None, normalize=True, already_normalized=False,
                              path_out="images/conf_matrix_normalized_row_no_augm_torchio.png")


############################
# MAIN
############################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Validation for Hippocampus Segmentation")
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
        "-W",
        "--write",
        default=False, type=bool,
        help="Boolean flag. Set to true for WRITE mode; false otherwise."
    )
    parser.add_argument(
        "--net",
        default='vnet',
        help="Specify the network to use [unet | vnet] ** FOR FUTURE RELEASES **"
    )

    args = parser.parse_args()
    run(logs_dir=args.dir, write_out=args.write, plot_conf=args.verbose)
