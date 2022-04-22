# Own Architectural Model for the Style Transfer that uses the ideas from GoogleNet, ResNet 
# After the feedback from Prof, we have decided to add many resnet layers and address the issue of checkerboard artifacts.
# The reason behind using the instance norm instead of batch norm (https://arxiv.org/abs/1607.08022) is that we would be able to train high-performance architectures 
# for real-time style transfer (web-app).

import torch 

class CustomModel(torch.nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

        # Aggresive downsampling, inspired by GoogleNet
        self.conv1 = ConvLayer(3, 64, kernel_size=7, stride=1)
        # we will use instance normalization instead of batch normalization, inspired by https://arxiv.org/abs/1607.08022
        self.norm1 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv2 = ConvLayer(64, 64, kernel_size=3, stride=2)
        self.norm2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.norm3 = torch.nn.InstanceNorm2d(128, affine=True)

        # We will add ResNet blocks to the network, inspired by ResNet paper
        self.res1 = ResNet(128)
        self.res2 = ResNet(128)
        self.res3 = ResNet(128)
        self.res4 = ResNet(128)
        self.res5 = ResNet(128)
        self.res6 = ResNet(128)
        self.res7 = ResNet(128)

        # Upsampling, inspired by https://distill.pub/2016/deconv-checkerboard/
        self.deconv1 = UpSampleConv(128, 64, kernel_size=3, stride=1, upsample=2)
        self.norm4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpSampleConv(64, 64, kernel_size=3, stride=1, upsample=2)
        self.norm5 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv3 = UpSampleConv(64, 32, kernel_size=3, stride=1, upsample=2)
        self.norm6 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv4 = UpSampleConv(32, 3, kernel_size=3, stride=1, upsample=2)

        # ReLU activation function
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.relu(self.norm3(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.relu(self.norm4(self.deconv1(x)))
        x = self.relu(self.norm5(self.deconv2(x)))
        x = self.relu(self.norm6(self.deconv3(x)))
        x = self.deconv4(x)
        return x



class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        # we have added reflection padding to add padding around the image
        self.ref_pad = torch.nn.ReflectionPad2d(kernel_size // 2)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        # get the padded input
        out = self.ref_pad(x)
        # get the output of the convolutional layer
        out = self.conv2d(out)
        # return the output
        return out

class ResNet(torch.nn.Module):
    """ResNet Block
    Inspired by ResNet paper
    """
    def __init__(self, channels):
        super(ResNet, self).__init__()
        # first convolutional layer
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        # we will use instance normalization instead of batch normalization, inspired by https://arxiv.org/abs/1607.08022
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        # second convolutional layer
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        # we will use instance normalization instead of batch normalization, inspired by https://arxiv.org/abs/1607.08022
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        # ReLU activation function
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # store x as a residual
        residual = x
        # get the output of the first convolutional layer
        out = self.relu(self.in1(self.conv1(x)))
        # get the output of the second convolutional layer
        out = self.in2(self.conv2(out))
        # add the residual to the output
        out = out + residual
        # return the output
        return out

class UpSampleConv(torch.nn.Module):
    '''
    Inspired from http://distill.pub/2016/deconv-checkerboard/
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpSampleConv, self).__init__()
        # get upsample 
        self.upsample = upsample
        # we will be using reflection padding to add zeros around the image
        self.ref_pad = torch.nn.ReflectionPad2d(kernel_size // 2)
        # define the convolutional layer
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        input_x = x # save the input
        # upsample the input
        if self.upsample:
            # interpolate the input using nearest neighbor interpolation
            input_x = torch.nn.functional.interpolate(input_x, mode='nearest', scale_factor=self.upsample)
        # get the padded input
        padded_x = self.ref_pad(input_x)
        # get the output of the convolutional layer
        output_x = self.conv(padded_x)
        # return the output
        return output_x