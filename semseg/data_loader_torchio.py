import torch
import os
import torchio
from torchio import Image, ImagesDataset

from semseg.data_loader import SemSegConfig


def TorchIODataLoader3DTraining(config: SemSegConfig) -> torch.utils.data.DataLoader:
    print('Building TorchIO Training Set Loader...')
    subject_list = list()
    for idx, (image_path, label_path) in enumerate(zip(config.train_images, config.train_labels)):
        s1 = torchio.Subject(
            t1=Image(type=torchio.INTENSITY, path=image_path),
            label=Image(type=torchio.LABEL, path=label_path),
        )

        subject_list.append(s1)

    subjects_dataset = ImagesDataset(subject_list, transform=config.transform_train)
    train_data = torch.utils.data.DataLoader(subjects_dataset, batch_size=config.batch_size,
                                             shuffle=True, num_workers=config.num_workers)
    print('TorchIO Training Loader built!')
    return train_data


def TorchIODataLoader3DValidation(config: SemSegConfig) -> torch.utils.data.DataLoader:
    print('Building TorchIO Validation Set Loader...')
    subject_list = list()
    for idx, (image_path, label_path) in enumerate(zip(config.val_images, config.val_labels)):
        s1 = torchio.Subject(
            t1=Image(type=torchio.INTENSITY, path=image_path),
            label=Image(type=torchio.LABEL, path=label_path),
        )

        subject_list.append(s1)

    subjects_dataset = ImagesDataset(subject_list, transform=config.transform_val)
    val_data = torch.utils.data.DataLoader(subjects_dataset, batch_size=config.batch_size,
                                           shuffle=False, num_workers=config.num_workers)
    print('TorchIO Validation Loader built!')
    return val_data


def get_pad_3d_image(pad_ref: tuple = (64, 64, 64), zero_pad: bool = True):
    def pad_3d_image(image):
        if zero_pad:
            value_to_pad = 0
        else:
            value_to_pad = image.min()
        pad_ref_channels = (image.shape[0], *pad_ref)
        # print("image.shape = {}".format(image.shape))
        if value_to_pad == 0:
            image_padded = torch.zeros(pad_ref_channels)
        else:
            image_padded = value_to_pad * torch.ones(pad_ref_channels)
        image_padded[:,:image.shape[1],:image.shape[2],:image.shape[3]] = image
        # print("image_padded.shape = {}".format(image_padded.shape))
        return image_padded
    return pad_3d_image


def z_score_normalization(inputs):
    input_mean = torch.mean(inputs)
    input_std = torch.std(inputs)
    return (inputs - input_mean)/input_std
