##########################
# Nicola Altini (2020)
# V-Net for Hippocampus Segmentation from MRI with PyTorch
##########################

##########################
# Imports
##########################
import torch
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from sklearn.model_selection import KFold

##########################
# Local Imports
##########################
from config.config import *
from config.paths import train_images_folder, train_labels_folder, train_prediction_folder, train_images, train_labels, \
    test_images_folder, test_images, test_prediction_folder
from semseg.utils import multi_dice_coeff, plot_confusion_matrix, zero_pad_3d_image, z_score_normalization
from run.utils import train_val_split
from sklearn.metrics import confusion_matrix, f1_score

##########################
# Config
##########################
config = SemSegMRIConfig()
attributes_config = [attr for attr in dir(config)
                     if not attr.startswith('__')]
print("Test Config")
for item in attributes_config:
    print("{:15s} ==> {}".format(item, getattr(config, item)))

###########################
# Load Net
###########################
cuda_dev = torch.device("cuda")

# Load From State Dict
# path_net = "logs/model_epoch_0080.pht"
# net = VNet3D(num_outs=config.num_outs, channels=config.num_channels)
# net.load_state_dict(torch.load(path_net))

path_net = "logs/model.pt"
path_nets_crossval = ["logs/model_folder_{:d}.pt".format(idx) for idx in range(config.num_folders)]

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
train_confusion_matrix = np.zeros((config.num_outs, config.num_outs))

for train_or_test_images, train_or_test_images_folder, train_or_test_prediction_folder, is_training in \
        zip(train_and_test_images, train_and_test_images_folder, train_and_test_prediction_folder, train_and_test):
    print("Images Folder: {}".format(train_or_test_images_folder))
    print("IsTraining: {}".format(is_training))

    kf = KFold(n_splits=config.num_folders)
    for idx_crossval, (train_index, val_index) in enumerate(kf.split(train_images)):
        if is_training:
            print("+==================+")
            print("+ Cross Validation +")
            print("+     Folder {:d}     +".format(idx_crossval))
            print("+==================+")
            print("TRAIN [Images: {:3d}]:\n{}".format(len(train_index), train_index))
            print("VAL   [Images: {:3d}]:\n{}".format(len(val_index), val_index))
            model_path = path_nets_crossval[idx_crossval]
            print("Model: {}".format(model_path))
            net = torch.load(model_path)
            _, train_or_test_images, _, train_labels_crossval = \
                train_val_split(train_images, train_labels, train_index, val_index, do_join=False)
        else:
            print("+============+")
            print("+   Test     +")
            print("+============+")
            net = torch.load(path_net)
        net = net.cuda(cuda_dev)
        net.eval()

        for idx, train_image in enumerate(train_or_test_images):
            print("Iter {} on {}".format(idx,len(train_or_test_images)))
            print("Image: {}".format(train_image))
            train_image_path = os.path.join(train_or_test_images_folder, train_image)

            if use_nib:
                train_image_nii = nib.load(str(train_image_path), mmap=False)
                train_image_np = train_image_nii.get_fdata(dtype=np.float32)
                affine = train_image_nii.affine
            else:
                train_image_sitk = sitk.ReadImage(train_image_path)
                train_image_np = sitk.GetArrayFromImage(train_image_sitk)
                origin, spacing, direction = train_image_sitk.GetOrigin(), \
                                             train_image_sitk.GetSpacing(), train_image_sitk.GetDirection()

                # print("Origin        {}".format(origin))
                # print("Spacing       {}".format(spacing))
                # print("Direction     {}".format(direction))
            train_image_np = z_score_normalization(train_image_np)

            inputs_padded = zero_pad_3d_image(train_image_np, pad_ref,
                                              value_to_pad=train_image_np.min())
                                                                       #         Z x Y x X
            inputs_padded = np.expand_dims(inputs_padded,axis=0)       #     1 x Z x Y x X
            inputs_padded = np.expand_dims(inputs_padded,axis=0)       # 1 x 1 x Z x Y x X

            with torch.no_grad():
                inputs = torch.from_numpy(inputs_padded).float()
                inputs = inputs.to(cuda_dev)
                outputs = net(inputs) # 1 x K x Z x Y x X
                outputs = torch.argmax(outputs, dim=1) # 1 x Z x Y x X
                outputs_np = outputs.data.cpu().numpy()

            outputs_np = outputs_np[0] # Z x Y x X
            outputs_np = outputs_np[:train_image_np.shape[0],
                                    :train_image_np.shape[1],
                                    :train_image_np.shape[2]]
            outputs_np = outputs_np.astype(np.uint8)
            filename_out = os.path.join(train_or_test_prediction_folder, train_image)
            if use_nib:
                outputs_nib = nib.Nifti1Image(outputs_np, affine)
                outputs_nib.header['qform_code'] = 1
                outputs_nib.header['sform_code'] = 0
                outputs_nib.to_filename(filename_out)
            else:
                outputs_sitk = sitk.GetImageFromArray(outputs_np)
                outputs_sitk.SetDirection(direction)
                outputs_sitk.SetSpacing(spacing)
                outputs_sitk.SetOrigin(origin)
                # print("Sum Outputs = {}".format(outputs_np.sum()))
                sitk.WriteImage(outputs_sitk, filename_out)

            if is_training:
                train_label = train_labels_crossval[idx]
                train_label_path = os.path.join(train_labels_folder, train_label)
                if use_nib:
                    train_label_nii = nib.load(str(train_label_path), mmap=False)
                    train_label_np = train_label_nii.get_fdata(dtype=np.float32)
                else:
                    train_label_sitk = sitk.ReadImage(train_label_path)
                    train_label_np = sitk.GetArrayFromImage(train_label_sitk)

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

multi_dices_np = np.array(multi_dices)
mean_multi_dice = np.mean(multi_dices_np)
std_multi_dice  = np.std(multi_dices_np,ddof=1)

f1_scores = np.array(f1_scores)

f1_scores_anterior_mean = np.mean(f1_scores[:,1])
f1_scores_anterior_std = np.std(f1_scores[:,1],ddof=1)

f1_scores_posterior_mean = np.mean(f1_scores[:,2])
f1_scores_posterior_std = np.std(f1_scores[:,2],ddof=1)

print("+================================+")
print("Multi Class Dice           ===> {:.4f} +/- {:.4f}".format(mean_multi_dice, std_multi_dice))
print("Images with Dice > 0.8     ===> {} on {}".format((multi_dices_np>0.8).sum(),multi_dices_np.size))
print("+================================+")
print("Hippocampus Anterior Dice  ===> {:.4f} +/- {:.4f}".format(f1_scores_anterior_mean, f1_scores_anterior_std))
print("Hippocampus Posterior Dice ===> {:.4f} +/- {:.4f}".format(f1_scores_posterior_mean, f1_scores_posterior_std))
print("+================================+")
print("Confusion Matrix")
print(train_confusion_matrix)
print("+================================+")
print("Normalized (All) Confusion Matrix")
train_confusion_matrix_normalized_all = train_confusion_matrix/train_confusion_matrix.sum()
print(train_confusion_matrix_normalized_all)
print("+================================+")
print("Normalized (Row) Confusion Matrix")
train_confusion_matrix_normalized_row = train_confusion_matrix.astype('float') / \
                                        train_confusion_matrix.sum(axis=1)[:, np.newaxis]
print(train_confusion_matrix_normalized_row)
print("+================================+")
plot_confusion_matrix(train_confusion_matrix,
                      target_names=None, title='Cross-Validation Confusion matrix',
                      cmap=None, normalize=False, already_normalized=False,
                      path_out="images/conf_matrix_no_norm_no_augm_torchio.png")
plot_confusion_matrix(train_confusion_matrix,
                      target_names=None, title='Cross-Validation Confusion matrix (row-normalized)',
                      cmap=None, normalize=True, already_normalized=False,
                      path_out="images/conf_matrix_normalized_row_no_augm_torchio.png")
