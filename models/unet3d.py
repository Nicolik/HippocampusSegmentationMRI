import torch
import torch.nn as nn
import torch.nn.functional as F

# Tensors for 3D Image Processing in PyTorch
# B x C x Z x Y x X
# Batch size BY Number of channels BY Z dim BY Y dim BY X dim


def InitialConvolution(in_channels, middle_channels, out_channels):
    # This is a block with 2 convolutions
    # The first convolution goes from in_channels to middle_channels feature maps
    # The second convolution goes from middle_channels to out_channels feature maps
    conv = nn.Sequential(
        nn.Conv3d(in_channels, middle_channels, stride=1, kernel_size=3, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv3d(middle_channels, out_channels, stride=1, kernel_size=3, padding=1),
        nn.ReLU(inplace=False)
    )
    return conv


def FinalConvolution(in_channels, out_channels):
    return nn.Conv3d(in_channels, out_channels, kernel_size=1)


def DownConvolution(in_channels, out_channels):
    # This is a block with 2 convolutions
    # The first convolution goes from in_channels to in_channels feature maps
    # The second convolution goes from in_channels to out_channels feature maps
    conv = nn.Sequential(
        nn.Conv3d(in_channels, in_channels, stride=1, kernel_size=3, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv3d(in_channels, out_channels, stride=1, kernel_size=3, padding=1),
        nn.ReLU(inplace=False)
    )
    return conv


def UpConvolution(in_channels, out_channels):
    # This is a block with 2 convolutions
    # The first convolution goes from in_channels to out_channels feature maps
    # The second convolution goes from out_channels to out_channels feature maps
    conv = nn.Sequential(
        nn.Conv3d(in_channels, out_channels, stride=1, kernel_size=3, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv3d(out_channels, out_channels, stride=1, kernel_size=3, padding=1),
        nn.ReLU(inplace=False)
    )
    return conv


def UpSample(in_channels, out_channels):
    # It doubles the spatial dimensions on every axes (x,y,z)
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)


def DownSample():
    # It halves the spatial dimensions on every axes (x,y,z)
    return nn.MaxPool3d(kernel_size=2, stride=2)


def CatBlock(x1, x2):
    return torch.cat((x1, x2), 1)


class UNet3D(nn.Module):
    def __init__(self, num_out_classes=2, input_channels=1, init_feat_channels=32):
        super().__init__()

        # Encoder layers definitions
        self.down_sample = DownSample()
        self.init_conv = InitialConvolution(input_channels, init_feat_channels, init_feat_channels*2)
        self.down_conv1 = DownConvolution(init_feat_channels*2, init_feat_channels*4)
        self.down_conv2 = DownConvolution(init_feat_channels*4, init_feat_channels*8)
        self.down_conv3 = DownConvolution(init_feat_channels*8, init_feat_channels*16)

        # Decoder layers definitions
        self.up_sample1 = UpSample(init_feat_channels*16, init_feat_channels*16)
        self.up_conv1   = UpConvolution(init_feat_channels*(16+8), init_feat_channels*8)
        self.up_sample2 = UpSample(init_feat_channels*8, init_feat_channels*8)
        self.up_conv2   = UpConvolution(init_feat_channels*(8+4), init_feat_channels*4)
        self.up_sample3 = UpSample(init_feat_channels*4, init_feat_channels*4)
        self.up_conv3   = UpConvolution(init_feat_channels*(4+2), init_feat_channels*2)
        self.final_conv = FinalConvolution(init_feat_channels*2, num_out_classes)

        # Softmax
        self.softmax = F.softmax

    def forward(self, image):
        # Encoder Part
        # B x  1 x Z x Y x X
        layer_init = self.init_conv(image)
        # B x 64 x Z x Y x X
        max_pool1  = self.down_sample(layer_init)
        # B x 64 x Z//2 x Y//2 x X//2

        layer_down2 = self.down_conv1(max_pool1)
        # B x 128 x Z//2 x Y//2 x X//2
        max_pool2   = self.down_sample(layer_down2)
        # B x 128 x Z//4 x Y//4 x X//4

        layer_down3 = self.down_conv2(max_pool2)
        # B x 256 x Z//4 x Y//4 x X//4
        max_pool_3  = self.down_sample(layer_down3)
        # B x 256 x Z//8 x Y//8 x X//8

        layer_down4 = self.down_conv3(max_pool_3)
        # B x 512 x Z//8 x Y//8 x X//8

        # Decoder part
        layer_up1 = self.up_sample1(layer_down4)
        # B x 512 x Z//4 x Y//4 x X//4
        cat_block1 = CatBlock(layer_down3, layer_up1)
        # B x (256+512) x Z//4 x Y//4 x X//4
        layer_conv_up1 = self.up_conv1(cat_block1)
        # B x 256 x Z//4 x Y//4 x X//4

        layer_up2 = self.up_sample2(layer_conv_up1)
        # B x 256 x Z//2 x Y//2 x X//2
        cat_block2 = CatBlock(layer_down2, layer_up2)
        # B x (128+256) x Z//2 x Y//2 x X//2
        layer_conv_up2 = self.up_conv2(cat_block2)
        # B x 128 x Z//2 x Y//2 x X//2

        layer_up3 = self.up_sample3(layer_conv_up2)
        # B x 128 x Z x Y x X
        cat_block3 = CatBlock(layer_init, layer_up3)
        # B x (64+128) x Z x Y x X
        layer_conv_up3 = self.up_conv3(cat_block3)
        # B x 64 x Z x Y x X
        final_layer = self.final_conv(layer_conv_up3)
        # B x 2 x Z x Y x X
        return self.softmax(final_layer, dim=1)


if __name__ == '__main__':

    unet = UNet3D(num_out_classes=2, input_channels=1, init_feat_channels=32)

    # B x C x Z x Y x X
    # 4 x 1 x 64 x 64 x 64
    input_batch_size = (4, 1, 64, 64, 64)
    input_example = torch.rand(input_batch_size)

    unet = unet.cuda()
    input_example = input_example.cuda()
    output = unet(input_example)
    # output = output.cpu().detach().numpy()
    # Expected output shape
    # B x N x Z x Y x X
    # 4 x 2 x 64 x 64 x 64
    expected_output_shape = (4, 2, 64, 64, 64)
    print("Output shape = {}".format(output.shape))
    assert output.shape == expected_output_shape, "Unexpected output shape, check the architecture!"

    expected_gt_shape = (4, 64, 64, 64)
    ground_truth = torch.ones(expected_gt_shape)
    ground_truth = ground_truth.long().cuda()

    # Defining loss fn
    ce_layer = torch.nn.CrossEntropyLoss()
    # Calculating loss
    ce_loss = ce_layer(output, ground_truth)
    print("CE Loss = {}".format(ce_loss))
    # Back propagation
    ce_loss.backward()
