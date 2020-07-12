from torchio import ZNormalization, Compose

from augm.lambda_channel import LambdaChannel
from semseg.data_loader import get_pad_3d_image

train_transforms_dict = {
    ZNormalization(): 1.0,
    # RandomAffine(): 0.05,
    # RandomElasticDeformation(max_displacement=3): 0.20,
    # RandomNoise(std=(0,0.1)): 0.10,
    # RandomBlur(std=(0,0.1)): 0.10,
    LambdaChannel(get_pad_3d_image(pad_ref=(48, 64, 48),zero_pad=False)): 1.0,
}
train_transform = Compose(train_transforms_dict)
val_transforms_dict = {
    ZNormalization(): 1.0,
    LambdaChannel(get_pad_3d_image(pad_ref=(48, 64, 48),zero_pad=False)): 1.0,
}
val_transform = Compose(val_transforms_dict)