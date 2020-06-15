##########################
# Nicola Altini (2020)
# V-Net for Hippocampus Segmentation from MRI with PyTorch
##########################

##########################
# Imports
##########################
import torch
import os
import numpy as np
import SimpleITK as sitk

##########################
# Local Imports
##########################
from config import *
from semseg.data_loader import min_max_normalization, zero_pad_3d_image, z_score_normalization
from semseg.utils import multi_dice_coeff
from models.vnet3d import VNet3D

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
use_final_model = True
if use_final_model:
    path_net = "logs/model.pt"
    net = torch.load(path_net)
else:
    path_net = "logs/model_epoch_0016.pht"
    net = VNet3D(num_outs=config.num_outs, channels=8)
    net.load_state_dict(torch.load(path_net))
net = net.cuda(cuda_dev)
net.eval()

###########################
# Eval Loop
###########################
pad_ref = (48,64,48)
multi_dices = list()

for idx, (train_image, train_label) in enumerate(zip(train_images, train_labels)):
    print("Iter {} on {}".format(idx,len(train_images)))
    print("Image: {}".format(train_image))
    train_image_path = os.path.join(train_images_folder, train_image)
    train_label_path = os.path.join(train_labels_folder, train_label)

    train_image_sitk = sitk.ReadImage(train_image_path)
    train_image_np = sitk.GetArrayFromImage(train_image_sitk)

    train_label_sitk = sitk.ReadImage(train_label_path)
    train_label_np = sitk.GetArrayFromImage(train_label_sitk)

    origin, spacing, direction = train_image_sitk.GetOrigin(), \
                                 train_image_sitk.GetSpacing(), train_image_sitk.GetDirection()
    print("Origin        {}".format(origin))
    print("Spacing       {}".format(spacing))
    print("Direction     {}".format(direction))

    # train_image_np = min_max_normalization(train_image_np)
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
    outputs_sitk = sitk.GetImageFromArray(outputs_np)
    outputs_sitk.SetDirection(direction)
    outputs_sitk.SetSpacing(spacing)
    outputs_sitk.SetOrigin(origin)
    print("Sum Outputs = {}".format(outputs_np.sum()))
    filename_out = os.path.join(train_prediction_folder, train_image)
    sitk.WriteImage(outputs_sitk, filename_out)

    multi_dice = multi_dice_coeff(np.expand_dims(train_label_np,axis=0),
                                  np.expand_dims(outputs_np,axis=0),
                                  config.num_outs)
    print("Multi Class Dice Coeff = {:.4f}".format(multi_dice))
    multi_dices.append(multi_dice)

multi_dices_np = np.array(multi_dices)
mean_multi_dice = np.mean(multi_dices_np)
std_multi_dice  = np.std(multi_dices_np,ddof=1)

print("Multi Class Dice       ===> {:.4f} +/- {:.4f}".format(mean_multi_dice, std_multi_dice))
print("Images with Dice > 0.8 ===> {} on {}".format((multi_dices_np>0.8).sum(),multi_dices_np.size))
