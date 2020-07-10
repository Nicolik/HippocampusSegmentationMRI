import os
from imgaug import augmenters as iaa
from config.augm import train_transform, val_transform
from config.paths import train_images_folder, train_labels_folder, train_images, train_labels
from semseg.data_loader import SemSegConfig

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
