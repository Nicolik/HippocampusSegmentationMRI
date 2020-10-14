import torch
import torch.nn as nn

def DownConvolution(in_channels, out_channels):
    conv = nn.Sequential(nn.Conv3d(in_channels, out_channels,stride=1, kernel_size=3, padding=1),
                         nn.ReLU(inplace=False),
                         nn.Conv3d(out_channels, out_channels, stride=1, kernel_size=3, padding=1),
                         nn.ReLU(inplace=False))
    return conv


def UpConvolution(in_channels, out_channels):
    conv = nn.Sequential(nn.Conv3d(in_channels, out_channels,stride=1, kernel_size=3, padding=1),
                         nn.ReLU(inplace=False),
                         nn.Conv3d(out_channels, out_channels, stride=1, kernel_size=3, padding=1),
                         nn.ReLU(inplace=False))
    return conv

def UpTranspose(in_channels, out_channels):
    layer = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
    return layer


def MaxPool():
    pool = nn.MaxPool3d(kernel_size=2, stride=2)
    return pool


def cat_block(x1):
        cat = torch.cat((x1,x1), 1)
        return cat


def FinalConvolution(in_channels=64, out_channels=2):
    final_conv = nn.Conv3d(in_channels,out_channels,kernel_size=1)
    return final_conv


class UNet(nn.Module):
    def __init__(self, num_outs=2, num_channels=1):
        super().__init__()

        self.down_conv1 = DownConvolution(num_channels, num_channels * 64)
        self.down_conv2 = DownConvolution(num_channels * 64, num_channels * 128)
        self.down_conv3 = DownConvolution(num_channels * 128, num_channels * 256)
        self.down_conv4 = DownConvolution(num_channels * 256, num_channels * 512)
        self.down_conv5 = DownConvolution(num_channels * 512, num_channels * 1024)

        self.max_pool = MaxPool() #UN MAX POOL PER TUTTI

        self.up_trans1 = UpTranspose(num_channels * 1024, num_channels * 512)
        self.up_trans2 = UpTranspose(num_channels * 512, num_channels * 256)
        self.up_trans3 = UpTranspose(num_channels * 256, num_channels * 128)
        self.up_trans4 = UpTranspose(num_channels * 128, num_channels * 64)

        self.up_conv1 = UpConvolution(num_channels * 1024, num_channels * 512)
        self.up_conv2 = UpConvolution(num_channels * 512, num_channels * 256)
        self.up_conv3 = UpConvolution(num_channels * 256, num_channels * 128)
        self.up_conv4 = UpConvolution(num_channels * 128, num_channels * 64)

        self.final_conv = FinalConvolution(in_channels=num_channels*64, out_channels=num_outs)

    def forward(self, image):

        # Fase di encoder

        layer_down1 = self.down_conv1(image)
        max_pool1 = self.max_pool(layer_down1)

        layer_down2 = self.down_conv2(max_pool1)
        max_pool2 = self.max_pool(layer_down2)

        layer_down3 = self.down_conv3(max_pool2)
        max_pool3 = self.max_pool(layer_down3)

        layer_down4 = self.down_conv4(max_pool3)
        max_pool4 = self.max_pool(layer_down4)

        layer_down5 = self.down_conv5(max_pool4)

        # Fase di decoder

        layer_trans_up1 = self.up_trans1(layer_down5)
        cat_block1 = cat_block(layer_trans_up1)
        layer_conv_up1 = self.up_conv1(cat_block1)

        layer_trans_up2 = self.up_trans2(layer_conv_up1)
        cat_block2 = cat_block(layer_trans_up2)
        layer_conv_up2 = self.up_conv2(cat_block2)

        layer_trans_up3 = self.up_trans3(layer_conv_up2)
        cat_block3 = cat_block(layer_trans_up3)
        layer_conv_up3 = self.up_conv3(cat_block3)

        layer_trans_up4 = self.up_trans4(layer_conv_up3)
        cat_block4 = cat_block(layer_trans_up4)
        layer_conv_up4 = self.up_conv4(cat_block4)

        final_layer = self.final_conv(layer_conv_up4)
        # print(final_layer.size())

        return final_layer
