# VGG16 as style transfer model 
import torch
from torchvision import models

class VGG16(torch.nn.Module):
    def __init__(self, requires_grad=True):
        super(VGG16, self).__init__()
        # load the pre-trained model and get features
        # self.vgg = models.vgg16(pretrained=True).features

        # add all the layers of vgg16
        self.conv1_1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        # load the features of the first layer
        # self.conv1_1.weight.data = self.vgg[0].weight.data
        self.conv1_2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # load the features of the second layer
        # self.conv1_2.weight.data = self.vgg[2].weight.data
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # load the features of the third layer
        # self.conv2_1.weight.data = self.vgg[5].weight.data
        self.conv2_2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        # load the features of the fourth layer
        # self.conv2_2.weight.data = self.vgg[7].weight.data
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        # load the features of the fifth layer
        # self.conv3_1.weight.data = self.vgg[10].weight.data
        self.conv3_2 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # load the features of the sixth layer
        # self.conv3_2.weight.data = self.vgg[12].weight.data
        self.conv3_3 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # load the features of the seventh layer
        # self.conv3_3.weight.data = self.vgg[14].weight.data
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        # load the features of the eighth layer
        # self.conv4_1.weight.data = self.vgg[17].weight.data
        self.conv4_2 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        # load the features of the ninth layer
        # self.conv4_2.weight.data = self.vgg[19].weight.data
        self.conv4_3 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        # load the features of the tenth layer
        # self.conv4_3.weight.data = self.vgg[21].weight.data
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        # load the features of the eleventh layer
        # self.conv5_1.weight.data = self.vgg[24].weight.data
        self.conv5_2 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        # load the features of the twelfth layer
        # self.conv5_2.weight.data = self.vgg[26].weight.data
        self.conv5_3 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        # load the features of the thirteenth layer
        # self.conv5_3.weight.data = self.vgg[28].weight.data
        self.pool5 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.relu = torch.nn.ReLU()

        # Up-sampling layers
        self.deconv1 = UpSampleConv(512, 256, kernel_size=3, stride=1, upsample=2)
        self.deconv2 = UpSampleConv(256, 128, kernel_size=3, stride=1, upsample=2)
        self.deconv3 = UpSampleConv(128, 64, kernel_size=3, stride=1, upsample=2)
        self.deconv4 = UpSampleConv(64, 32, kernel_size=3, stride=1, upsample=2)
        self.deconv5 = UpSampleConv(32, 3, kernel_size=3, stride=1, upsample=2)

        # if gradients are required, update parameters to have gradients
        if requires_grad:
            for param in self.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.pool3(x)
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x = self.pool4(x)
        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        x = self.pool5(x)
        # upsample the features
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        return x




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


# Program End
    