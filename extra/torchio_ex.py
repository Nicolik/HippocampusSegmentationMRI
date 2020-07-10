import torch
import torchio
from torchio import ImagesDataset, Image, ZNormalization, Compose
from config.config import *
from config.paths import train_images_folder, train_labels_folder, train_images, train_labels
from semseg.data_loader_torchio import get_pad_3d_image
from augm.lambda_channel import LambdaChannel

transforms_dict = {
    ZNormalization(): 1.0,
    # LambdaChannel(z_score_normalization, types_to_apply=torchio.INTENSITY): 1.0,
    # RandomAffine(): 0.25,
    # RandomElasticDeformation(max_displacement=3): 0.25,
    LambdaChannel(get_pad_3d_image(pad_ref=(48, 64, 48), zero_pad=False)): 1.0,
}

transform = Compose(transforms_dict)

subject_list = list()

idx = 0
for idx,(train_image, train_label) in enumerate(zip(train_images, train_labels)):
    image_path = os.path.join(train_images_folder, train_image)
    label_path = os.path.join(train_labels_folder, train_label)

    s1 = torchio.Subject(
        t1    = Image(type=torchio.INTENSITY, path=image_path),
        label = Image(type=torchio.LABEL, path=label_path),
    )

    subject_list.append(s1)

subjects_dataset = ImagesDataset(subject_list, transform=transform)
subject_sample = subjects_dataset[0]

for idx in range(0,len(train_images[:10])):
    subject_sample = subjects_dataset[idx]
    print("Iter {} on {}".format(idx+1,len(train_images)))
    print("t1.shape       = {}".format(subject_sample.t1.shape))
    print("label.shape    = {}".format(subject_sample.label.shape))
    print("t1 [min - max] = [{:.1f} : {:.1f}]".format(subject_sample.t1.data.min(),subject_sample.t1.data.max()))
    print("label.unique   = {}".format(subject_sample.label.data.unique()))


config = SemSegMRIConfig()
train_data = torch.utils.data.DataLoader(subjects_dataset, batch_size=config.batch_size,
                                         shuffle=False, num_workers=config.num_workers)


iterable_data_loader = iter(train_data)
el = next(iterable_data_loader)
inputs = el['t1']['data']
labels = el['label']['data']
print("Shape of Batch: [input {}] [label {}]".format(inputs.shape, labels.shape))

for i, el in enumerate(train_data):
    print("Iteration {} on {}".format(i+1,len(train_data)))
    inputs = el['t1']['data']
    labels = el['label']['data']
    print("Shape of Batch: [input {}] [label {}]".format(inputs.shape, labels.shape))
    print("Range Inputs  : [{:.2f} %% {:.2f}]".format(inputs.min(),inputs.max()))
