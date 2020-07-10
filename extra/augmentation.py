import SimpleITK as sitk
import numpy as np
import torch
from config.config import *
from config.paths import train_images_folder, train_labels_folder, train_images, train_labels

transforms_dict = {
    RandomAffine(): 0.75,
    RandomElasticDeformation(max_displacement=3): 0.25,
}

transform = Compose(transforms_dict)

idx = 0
train_image = train_images[0]
train_label = train_labels[0]

image_path = os.path.join(train_images_folder, train_image)
label_path = os.path.join(train_labels_folder, train_label)

image_sitk = sitk.ReadImage(image_path)
label_sitk = sitk.ReadImage(label_path)

image_np = sitk.GetArrayFromImage(image_sitk)
label_np = sitk.GetArrayFromImage(label_sitk)

image_np = np.expand_dims(image_np,axis=0)
label_np = np.expand_dims(label_np,axis=0)

image_np_out = transform(image_np)

image_t = torch.from_numpy(image_np)
label_t = torch.from_numpy(label_np)

s1 = torchio.Subject(
    t1    = torchio.Image(type=torchio.INTENSITY, tensor=image_t),
    label = torchio.Image(type=torchio.LABEL, tensor=label_t),
)
out_t = transform(s1)

# +=============================+
# +  RandomElasticDeformation   +
# +=============================+

image_sitk.GetSize()
image_sitk.GetSpacing()
bounds = np.array(image_sitk.GetSize()) * np.array(image_sitk.GetSpacing())
num_control_points = np.array((7, 7, 6))
grid_spacing = bounds / (num_control_points - 2)
print("Grid Spacing = ", grid_spacing)
potential_folding = grid_spacing / 2
print("Potential Folding = ", potential_folding)
