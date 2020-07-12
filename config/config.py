import os
from config.augm import train_transform, val_transform
from config.paths import train_images_folder, train_labels_folder, train_images, train_labels
from semseg.data_loader import SemSegConfig


class SemSegMRIConfig(SemSegConfig):
    train_images = [os.path.join(train_images_folder, train_image)
                    for train_image in train_images]
    train_labels = [os.path.join(train_labels_folder, train_label)
                    for train_label in train_labels]
    val_images = None
    val_labels = None
    do_normalize = True
    batch_size = 12
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
    num_channels = 4
    transform_train = train_transform
    transform_val = val_transform
