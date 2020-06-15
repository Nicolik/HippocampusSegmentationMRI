import torch
import SimpleITK as sitk
import numpy as np


class SemSegConfig():
    train_images = None
    train_labels = None
    do_normalize = True
    augmentation = None
    zero_pad     = True
    pad_ref      = (64,64,64)
    batch_size   = 4
    num_workers  = 0


class Dataset3DFull(torch.utils.data.Dataset):
    def __init__(self, train_images, train_labels, augmentation=None,
                 do_normalize=True, zero_pad=True, pad_ref=(64,64,64)):
        self.train_images = train_images
        self.train_labels = train_labels
        self.augmentation = augmentation
        self.do_normalize = do_normalize
        self.pad_ref = pad_ref
        self.zero_pad = zero_pad
        assert len(train_images) == len(train_labels), \
            "Mismatch between number of training images and labels"

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, idx):

        image_filename = self.train_images[idx]
        label_filename = self.train_labels[idx]

        image_sitk = sitk.ReadImage(image_filename)
        image_np = sitk.GetArrayFromImage(image_sitk)

        label_sitk = sitk.ReadImage(label_filename)
        label_np = sitk.GetArrayFromImage(label_sitk)

        inputs, labels = image_np, label_np
        # DEBUG ONLY
        # print("Shapes: [image {}] [label {}]".format(inputs.shape, labels.shape))
        # print("Range - Before Normalization - [{:.1f} {:.1f}]".format(inputs.min(),inputs.max()))

        if self.do_normalize:
            inputs = self.normalize(inputs)
            # print("Range - After  Normalization - [{:.1f} {:.1f}]".format(inputs.min(), inputs.max()))
        if self.augmentation is not None:
            inputs, labels = self.perform_augmentation(inputs, labels)
            # print("Range - After  Augmentation  - [{:.1f} {:.1f}]".format(inputs.min(), inputs.max()))
        if self.zero_pad:
            inputs, labels = self.perform_zero_pad(inputs, labels, value_to_pad=inputs.min())
            # print("Range - After  Padding       - [{:.1f} {:.1f}]".format(inputs.min(), inputs.max()))

        inputs,labels = np.expand_dims(inputs,axis=0), np.expand_dims(labels,axis=0)

        features, targets = torch.from_numpy(inputs).float(), torch.from_numpy(labels).long()
        return (features, targets)

    def perform_augmentation(self, inputs, labels):
        # TODO: implement some augmentation technique
        return (inputs, labels)

    def normalize(self, inputs):
        return z_score_normalization(inputs)

    def perform_zero_pad(self, inputs, labels, value_to_pad = 0):
        return (zero_pad_3d_image(inputs, self.pad_ref, value_to_pad),
                zero_pad_3d_image(labels, self.pad_ref, 0))


def zero_pad_3d_image(image, pad_ref=(64,64,64), value_to_pad = 0):
    if value_to_pad == 0:
        image_padded = np.zeros(pad_ref)
    else:
        image_padded = value_to_pad * np.ones(pad_ref)
    image_padded[:image.shape[0],:image.shape[1],:image.shape[2]] = image
    return image_padded


def GetDataLoader3D(config: SemSegConfig) -> torch.utils.data.DataLoader:
    print('Building Training Set Loader...')
    train = Dataset3DFull(config.train_images, config.train_labels,
                          augmentation=config.augmentation, do_normalize=config.do_normalize,
                          zero_pad=config.zero_pad, pad_ref=config.pad_ref)

    train_data = torch.utils.data.DataLoader(train, batch_size=config.batch_size, shuffle=True,
                                             num_workers=config.num_workers)
    print('Training Loader built!')
    return train_data


def min_max_normalization(input):
    return (input - input.min()) / (input.max() - input.min())


def z_score_normalization(input):
    input_mean = np.mean(input)
    input_std = np.std(input)
    # print("Mean = {:.2f} - Std = {:.2f}".format(input_mean,input_std))
    return (input - input_mean)/input_std
