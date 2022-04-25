# Own Architectural Model for the Style Transfer that uses the ideas from GoogleNet, ResNet 
# After the feedback from Prof, we have decided to add many resnet layers and address the issue of checkerboard artifacts.
# The reason behind using the instance norm instead of batch norm (https://arxiv.org/abs/1607.08022) is that we would be able to train high-performance architectures 
# for real-time style transfer (web-app).

import torch 

# class CustomModel(torch.nn.Module):
#     def __init__(self):
#         super(CustomModel, self).__init__()

#         # Aggresive downsampling, inspired by GoogleNet
#         self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
#         # we will use instance normalization instead of batch normalization, inspired by https://arxiv.org/abs/1607.08022
#         # self.norm1 = torch.nn.InstanceNorm2d(32, affine=True)
#         self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
#         # self.norm2 = torch.nn.InstanceNorm2d(64, affine=True)
#         self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
#         # self.norm3 = torch.nn.InstanceNorm2d(128, affine=True)

#         # We will add ResNet blocks to the network, inspired by ResNet paper
#         self.res1 = ResNet(128)
#         self.res2 = ResNet(128)
#         self.res3 = ResNet(128)
#         self.res4 = ResNet(128)
#         self.res5 = ResNet(128)
#         # self.res6 = ResNet(128)
#         # self.res7 = ResNet(128)

#         # Upsampling, inspired by https://distill.pub/2016/deconv-checkerboard/
#         self.deconv1 = UpSampleConv(128, 64, kernel_size=3, stride=1, upsample=2)
#         # self.norm4 = torch.nn.InstanceNorm2d(64, affine=True)
#         self.deconv2 = UpSampleConv(64, 32, kernel_size=3, stride=1, upsample=2)
#         # self.norm5 = torch.nn.InstanceNorm2d(32, affine=True)
#         # self.deconv3 = UpSampleConv(64, 32, kernel_size=3, stride=1, upsample=2)
#         # self.norm6 = torch.nn.InstanceNorm2d(32, affine=True)
#         self.deconv4 = ConvLayer(32, 3, kernel_size=9, stride=1)

#         # ReLU activation function
#         self.relu = torch.nn.ReLU()

#     def forward(self, x):
#         x = self.relu(self.norm1(self.conv1(x)))
#         x = self.relu(self.norm2(self.conv2(x)))
#         x = self.relu(self.norm3(self.conv3(x)))
#         x = self.res1(x)
#         x = self.res2(x)
#         x = self.res3(x)
#         x = self.res4(x)
#         x = self.res5(x)
#         # x = self.res6(x)
#         # x = self.res7(x)
#         x = self.relu(self.norm4(self.deconv1(x)))
#         x = self.relu(self.norm5(self.deconv2(x)))
#         # x = self.relu(self.norm6(self.deconv3(x)))
#         x = self.deconv4(x)
#         return x


class CustomModel(torch.nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.ConvBlock = torch.nn.Sequential(
            ConvLayer(3, 32, 9, 1),
            torch.nn.ReLU(),
            ConvLayer(32, 64, 3, 2),
            torch.nn.ReLU(),
            ConvLayer(64, 128, 3, 2),
            torch.nn.ReLU()
        )
        self.ResidualBlock = torch.nn.Sequential(
            ResNet(128, 3), 
            ResNet(128, 3), 
            ResNet(128, 3), 
            ResNet(128, 3), 
            ResNet(128, 3)
        )
        self.DeconvBlock = torch.nn.Sequential(
            UpSampleConv(128, 64, 3, 2, 1),
            torch.nn.ReLU(),
            UpSampleConv(64, 32, 3, 2, 1),
            torch.nn.ReLU(),
            ConvLayer(32, 3, 9, 1, norm="None")
        )

    def forward(self, x):
        x = self.ConvBlock(x)
        x = self.ResidualBlock(x)
        out = self.DeconvBlock(x)
        return out

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm = 'instance'):
        super(ConvLayer, self).__init__()
        # we have added reflection padding to add padding around the image
        self.ref_pad = torch.nn.ReflectionPad2d(kernel_size // 2)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        # Normalization Layers
        self.norm_type = norm
        if (norm=="instance"):
            self.norm_layer = torch.nn.InstanceNorm2d(out_channels, affine=True)
        elif (norm=="batch"):
            self.norm_layer = torch.nn.BatchNorm2d(out_channels, affine=True)


    def forward(self, x):
        # get the padded input
        x = self.ref_pad(x)
        # get the output of the convolutional layer
        x = self.conv2d(x)
        if (self.norm_type=="None"):
            out = x
        else:
            out = self.norm_layer(x)
        # return the output
        return out

class ResNet(torch.nn.Module):
    """ResNet Block
    Inspired by ResNet paper
    """
    def __init__(self, channels, kernel_size=3):
        super(ResNet, self).__init__()
        # first convolutional layer
        self.conv1 = ConvLayer(channels, channels, kernel_size, stride=1)

        # # we will use instance normalization instead of batch normalization, inspired by https://arxiv.org/abs/1607.08022
        # self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)

        # second convolutional layer
        self.conv2 = ConvLayer(channels, channels, kernel_size, stride=1)

        # # we will use instance normalization instead of batch normalization, inspired by https://arxiv.org/abs/1607.08022
        # self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)

        # ReLU activation function
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # store x as a residual
        residual = x
        # get the output of the first convolutional layer
        out = self.relu(self.conv1(x))
        # get the output of the second convolutional layer
        out = self.conv2(out)
        # add the residual to the output
        out = out + residual
        # return the output
        return out

class UpSampleConv(torch.nn.Module):
    '''
    Inspired from http://distill.pub/2016/deconv-checkerboard/
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding, norm = 'instance'):
        super(UpSampleConv, self).__init__()

        # # get upsample 
        # self.upsample = upsample

        # # we will be using reflection padding to add zeros around the image
        # self.ref_pad = torch.nn.ReflectionPad2d(kernel_size // 2)
        # # define the convolutional layer
        # self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        # Transposed Convolution 
        padding_size = kernel_size // 2
        self.conv_transpose = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding_size, output_padding)

        # Normalization Layers
        self.norm_type = norm
        if (norm=="instance"):
            self.norm_layer = torch.nn.InstanceNorm2d(out_channels, affine=True)
        elif (norm=="batch"):
            self.norm_layer = torch.nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        # input_x = x # save the input
        # # upsample the input
        # if self.upsample:
        #     # interpolate the input using nearest neighbor interpolation
        #     input_x = torch.nn.functional.interpolate(input_x, mode='nearest', scale_factor=self.upsample)
        # # get the padded input
        # padded_x = self.ref_pad(input_x)
        # # get the output of the convolutional layer
        # output_x = self.conv(padded_x)
        # # return the output
        # return output_x

        x = self.conv_transpose(x)
        if (self.norm_type=="None"):
            out = x
        else:
            out = self.norm_layer(x)
        return out