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
from semseg.data_loader import min_max_normalization
from models.vnet3d import VXNet3D

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
    net = VXNet3D(num_outs=config.num_outs, channels=8)
    net.load_state_dict(torch.load(path_net))
net = net.cuda(cuda_dev)
net.eval()

###########################
# Eval Loop
###########################
pad_ref = (64,64,64)

for idx, train_image in enumerate(train_images):
    print("Iter {} on {}".format(idx,len(train_images)))
    print("Image: {}".format(train_image))
    train_image_path = os.path.join(train_images_folder, train_image)

    train_image_sitk = sitk.ReadImage(train_image_path)
    train_image_np = sitk.GetArrayFromImage(train_image_sitk)

    origin, spacing, direction = train_image_sitk.GetOrigin(), \
                                 train_image_sitk.GetSpacing(), train_image_sitk.GetDirection()
    print("Origin        {}".format(origin))
    print("Spacing       {}".format(spacing))
    print("Direction     {}".format(direction))

    train_image_np = min_max_normalization(train_image_np)

    inputs_padded = np.zeros(pad_ref) # Z x Y x X
    inputs_padded[:train_image_np.shape[0],
                  :train_image_np.shape[1],
                  :train_image_np.shape[2]] = train_image_np

    inputs_padded = np.expand_dims(inputs_padded,axis=0) #     1 x Z x Y x X
    inputs_padded = np.expand_dims(inputs_padded,axis=0) # 1 x 1 x Z x Y x X

    with torch.no_grad():
        inputs = torch.from_numpy(inputs_padded).float()
        inputs = inputs.to(cuda_dev)
        outputs = net(inputs) # 1 x K x Z x Y x X
        outputs = torch.argmax(outputs, dim=1) # 1 x Z x Y x X
        outputs_np = outputs.data.cpu().numpy()

    outputs_np = outputs_np[0]
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
