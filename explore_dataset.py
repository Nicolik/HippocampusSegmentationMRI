import os
import numpy as np
import SimpleITK as sitk
from config import *
from semseg.utils import min_max_normalization

sizes_list = list()
min_list = list()
max_list = list()

for idx, train_image in enumerate(train_images):
    print("Iter {} on {}".format(idx+1, len(train_images)))
    print("Image: {}".format(train_image))
    t1_filename = os.path.join(train_images_folder, train_image)

    # Read the .nii image containing the volume with SimpleITK:
    sitk_t1 = sitk.ReadImage(t1_filename)

    # and access the numpy array:
    t1 = sitk.GetArrayFromImage(sitk_t1)

    sizes = t1.shape
    sizes_list.append(sizes)
    print("Sizes = {}".format(sizes))
    min_val = t1.min()
    max_val = t1.max()
    print("[Before Normalization]")
    print("Min = {:.1f} Max = {:.1f}".format(min_val, max_val))
    min_list.append(min_val)
    max_list.append(max_val)

    t1_norm = min_max_normalization(t1)
    min_val_norm = t1_norm.min()
    max_val_norm = t1_norm.max()
    print("[After  Normalization]")
    print("Min = {:.1f} Max = {:.1f}".format(min_val_norm, max_val_norm))

print("Min belongs to range {:.1f} - {:.1f}. Mean: {:.1f}".
      format(min(min_list),max(min_list),sum(min_list)/len(min_list)))
print("Max belongs to range {:.1f} - {:.1f}. Mean: {:.1f}".
      format(min(max_list),max(max_list),sum(max_list)/len(max_list)))

z_list, y_list, x_list = list(), list(), list()
for size in sizes_list:
    z,y,x = size
    z_list.append(z)
    y_list.append(y)
    x_list.append(x)

print("Z ranges from {} to {}".format(min(z_list),max(z_list)))
print("Y ranges from {} to {}".format(min(y_list),max(y_list)))
print("X ranges from {} to {}".format(min(x_list),max(x_list)))