import os
from imgaug import augmenters as iaa
import torchio
from torchio.transforms import (ZNormalization, RandomAffine, Compose,
                                RandomElasticDeformation, RandomNoise, RandomBlur)
from augm.lambda_channel import LambdaChannel
from semseg.data_loader_torchio import get_pad_3d_image, z_score_normalization
from semseg.data_loader import SemSegConfig

logs_folder = "logs"

base_dataset_dir = os.path.join("datasets","Task04_Hippocampus")

train_images_folder = os.path.join(base_dataset_dir, "imagesTr")
train_labels_folder = os.path.join(base_dataset_dir, "labelsTr")

train_prediction_folder = os.path.join(base_dataset_dir, "predTr")
os.makedirs(train_prediction_folder, exist_ok=True)

train_images = os.listdir(train_images_folder)
train_labels = os.listdir(train_labels_folder)

train_images = [train_image for train_image in train_images
                if train_image.endswith(".nii.gz")]
train_labels = [train_label for train_label in train_labels
                if train_label.endswith(".nii.gz")]

test_images_folder = os.path.join(base_dataset_dir, "imagesTs")
test_images = os.listdir(test_images_folder)

test_prediction_folder = os.path.join(base_dataset_dir, "predTs")
os.makedirs(test_prediction_folder, exist_ok=True)

labels_names = {
   "0": "background",
   "1": "Anterior",
   "2": "Posterior"
 }
labels_names_list = [labels_names[el] for el in labels_names]

augmentation_list = iaa.SomeOf((0,2), [
                iaa.GaussianBlur(sigma=(0.0, 0.1)),
                iaa.ElasticTransformation(alpha=(3,5), sigma=2.5),
                iaa.Multiply((0.98, 1.02)),
            ])


train_transforms_dict = {
    ZNormalization(): 1.0,
    RandomAffine(): 0.05,
    RandomElasticDeformation(max_displacement=3): 0.20,
    RandomNoise(std=(0,0.1)): 0.10,
    RandomBlur(std=(0,0.1)): 0.10,
    LambdaChannel(get_pad_3d_image(pad_ref=(48, 64, 48),zero_pad=False)): 1.0,
}
train_transform = Compose(train_transforms_dict)

val_transforms_dict = {
    ZNormalization(): 1.0,
    LambdaChannel(get_pad_3d_image(pad_ref=(48, 64, 48),zero_pad=False)): 1.0,
}
val_transform = Compose(val_transforms_dict)


class SemSegMRIConfig(SemSegConfig):
    train_images = [os.path.join(train_images_folder, train_image)
                    for train_image in train_images]
    train_labels = [os.path.join(train_labels_folder, train_label)
                    for train_label in train_labels]
    val_images = None
    val_labels = None
    do_normalize = True
    augmentation = None
    batch_size = 4
    num_workers = 0
    pad_ref = (48, 64, 48)
    lr = 0.01
    epochs = 100
    low_lr_epoch = epochs // 3
    val_epochs = epochs // 5
    cuda = True
    num_outs = 3
    do_crossval = True
    num_folders = 5
    num_channels = 8
    transform_train = train_transform
    transform_val = val_transform
