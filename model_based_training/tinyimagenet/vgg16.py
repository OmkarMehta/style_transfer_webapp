

# VGG16 as style transfer model 
import torch
from torchvision import models

class VGG16(torch.nn.Module):
    def __init__(self, vgg_path="models/vgg16-00b39a1b.pth"):
        super(VGG16, self).__init__()
        

        # load the model of vgg16 with pretrained set to False
        vgg16_features = models.vgg16(pretrained=False)
        # load the weights of vgg16 from umich, cited in the report
        vgg16_features.load_state_dict(torch.load(vgg_path), strict=False)
        # get the features of vgg16
        self.features = vgg16_features.features[:20]
        self.conv21 = vgg16_features.features[21]
        self.in21 = torch.nn.InstanceNorm2d(512, affine=True)
        self.conv24 = vgg16_features.features[24]
        self.in24 = torch.nn.InstanceNorm2d(512, affine=True)
        self.conv26 = vgg16_features.features[26]
        self.in26 = torch.nn.InstanceNorm2d(512, affine=True)
        self.conv28 = vgg16_features.features[28]
        self.in28 = torch.nn.InstanceNorm2d(512, affine=True)
        self.relu = torch.nn.ReLU()

        for name, param in vgg16_features.features.named_parameters():
            if param.requires_grad == True:
                if '26' in name or '28' in name or '24' in name or '21' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        self.DeConvBlock = torch.nn.Sequential(
            # UpSampleConv(512, 512, 3, 2, 1), # deconv5_1
            # torch.nn.ReLU(),
            UpSampleConv(512, 128, 3, 2, 1), # deconv4_1
            torch.nn.ReLU(),
            # UpSampleConv(256, 128, 3, 2, 1), # deconv3_1
            # torch.nn.ReLU(),
            UpSampleConv(128, 64, 3, 2, 1), # deconv2_1
            torch.nn.ReLU(),
            # UpSampleConv(64, 64, 3, 2, 1), # deconv2_2
            # torch.nn.ReLU(),
            UpSampleConv(64, 32, 3, 2, 1), # deconv1_1
            torch.nn.ReLU(),
            ConvLayer(32, 3, 7, 1, norm = 'None')  
        )             
    def forward(self, x):
        x = self.features(x)
        x = self.relu(self.in21(self.conv21(x)))
        x = self.relu(self.in24(self.conv24(x)))
        x = self.relu(self.in26(self.conv26(x)))
        x = self.relu(self.in28(self.conv28(x)))
        x = self.DeConvBlock(x)

        return x

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

