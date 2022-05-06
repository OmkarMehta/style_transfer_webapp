# # VGG16 as style transfer model 
# import torch
# from torchvision import models

# class VGG16(torch.nn.Module):
#     def __init__(self, requires_grad=True):
#         super(VGG16, self).__init__()
#         # load the pre-trained model and get features
#         # self.vgg = models.vgg16(pretrained=True).features

#         # add all the layers of vgg16
#         self.conv1_1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
#         # load the features of the first layer
#         # self.conv1_1.weight.data = self.vgg[0].weight.data
#         self.conv1_2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         # load the features of the second layer
#         # self.conv1_2.weight.data = self.vgg[2].weight.data
#         self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

#         self.conv2_1 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         # load the features of the third layer
#         # self.conv2_1.weight.data = self.vgg[5].weight.data
#         self.conv2_2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
#         # load the features of the fourth layer
#         # self.conv2_2.weight.data = self.vgg[7].weight.data
#         self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

#         self.conv3_1 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
#         # load the features of the fifth layer
#         # self.conv3_1.weight.data = self.vgg[10].weight.data
#         self.conv3_2 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         # load the features of the sixth layer
#         # self.conv3_2.weight.data = self.vgg[12].weight.data
#         self.conv3_3 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         # load the features of the seventh layer
#         # self.conv3_3.weight.data = self.vgg[14].weight.data
#         self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

#         self.conv4_1 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
#         # load the features of the eighth layer
#         # self.conv4_1.weight.data = self.vgg[17].weight.data
#         self.conv4_2 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#         # load the features of the ninth layer
#         # self.conv4_2.weight.data = self.vgg[19].weight.data
#         self.conv4_3 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#         # load the features of the tenth layer
#         # self.conv4_3.weight.data = self.vgg[21].weight.data
#         self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

#         self.conv5_1 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#         # load the features of the eleventh layer
#         # self.conv5_1.weight.data = self.vgg[24].weight.data
#         self.conv5_2 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#         # load the features of the twelfth layer
#         # self.conv5_2.weight.data = self.vgg[26].weight.data
#         self.conv5_3 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#         # load the features of the thirteenth layer
#         # self.conv5_3.weight.data = self.vgg[28].weight.data
#         self.pool5 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

#         self.relu = torch.nn.ReLU()

#         # Up-sampling layers
#         self.deconv1 = UpSampleConv(512, 256, kernel_size=3, stride=1, upsample=2)
#         self.deconv2 = UpSampleConv(256, 128, kernel_size=3, stride=1, upsample=2)
#         self.deconv3 = UpSampleConv(128, 64, kernel_size=3, stride=1, upsample=2)
#         self.deconv4 = UpSampleConv(64, 32, kernel_size=3, stride=1, upsample=2)
#         self.deconv5 = UpSampleConv(32, 3, kernel_size=3, stride=1, upsample=2)

#         # if gradients are required, update parameters to have gradients
#         if requires_grad:
#             for param in self.parameters():
#                 param.requires_grad = True
    
#     def forward(self, x):
#         x = self.relu(self.conv1_1(x))
#         x = self.relu(self.conv1_2(x))
#         x = self.pool1(x)
#         x = self.relu(self.conv2_1(x))
#         x = self.relu(self.conv2_2(x))
#         x = self.pool2(x)
#         x = self.relu(self.conv3_1(x))
#         x = self.relu(self.conv3_2(x))
#         x = self.relu(self.conv3_3(x))
#         x = self.pool3(x)
#         x = self.relu(self.conv4_1(x))
#         x = self.relu(self.conv4_2(x))
#         x = self.relu(self.conv4_3(x))
#         x = self.pool4(x)
#         x = self.relu(self.conv5_1(x))
#         x = self.relu(self.conv5_2(x))
#         x = self.relu(self.conv5_3(x))
#         x = self.pool5(x)
#         # upsample the features
#         x = self.deconv1(x)
#         x = self.deconv2(x)
#         x = self.deconv3(x)
#         x = self.deconv4(x)
#         x = self.deconv5(x)
#         return x




# class UpSampleConv(torch.nn.Module):
#     '''
#     Inspired from http://distill.pub/2016/deconv-checkerboard/
#     '''
#     def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
#         super(UpSampleConv, self).__init__()
#         # get upsample 
#         self.upsample = upsample
#         # we will be using reflection padding to add zeros around the image
#         self.ref_pad = torch.nn.ReflectionPad2d(kernel_size // 2)
#         # define the convolutional layer
#         self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

#     def forward(self, x):
#         input_x = x # save the input
#         # upsample the input
#         if self.upsample:
#             # interpolate the input using nearest neighbor interpolation
#             input_x = torch.nn.functional.interpolate(input_x, mode='nearest', scale_factor=self.upsample)
#         # get the padded input
#         padded_x = self.ref_pad(input_x)
#         # get the output of the convolutional layer
#         output_x = self.conv(padded_x)
#         # return the output
#         return output_x


# # Program End

# VGG16 as style transfer model 
import torch
from torchvision import models

class VGG16(torch.nn.Module):
    def __init__(self, vgg_path="models/vgg16-00b39a1b.pth"):
        super(VGG16, self).__init__()
        # # load the pre-trained model and get features
        # # self.vgg = models.vgg16(pretrained=True).features

        # # add all the layers of vgg16
        # self.conv1_1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        # # load the features of the first layer
        # # self.conv1_1.weight.data = self.vgg[0].weight.data
        # self.conv1_2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # # load the features of the second layer
        # # self.conv1_2.weight.data = self.vgg[2].weight.data
        # self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # self.conv2_1 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # # load the features of the third layer
        # # self.conv2_1.weight.data = self.vgg[5].weight.data
        # self.conv2_2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        # # load the features of the fourth layer
        # # self.conv2_2.weight.data = self.vgg[7].weight.data
        # self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # self.conv3_1 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        # # load the features of the fifth layer
        # # self.conv3_1.weight.data = self.vgg[10].weight.data
        # self.conv3_2 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # # load the features of the sixth layer
        # # self.conv3_2.weight.data = self.vgg[12].weight.data
        # self.conv3_3 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # # load the features of the seventh layer
        # # self.conv3_3.weight.data = self.vgg[14].weight.data
        # self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # self.conv4_1 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        # # load the features of the eighth layer
        # # self.conv4_1.weight.data = self.vgg[17].weight.data
        # self.conv4_2 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        # # load the features of the ninth layer
        # # self.conv4_2.weight.data = self.vgg[19].weight.data
        # self.conv4_3 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        # # load the features of the tenth layer
        # # self.conv4_3.weight.data = self.vgg[21].weight.data
        # self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # self.conv5_1 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        # # load the features of the eleventh layer
        # # self.conv5_1.weight.data = self.vgg[24].weight.data
        # self.conv5_2 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        # # load the features of the twelfth layer
        # # self.conv5_2.weight.data = self.vgg[26].weight.data
        # self.conv5_3 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        # # load the features of the thirteenth layer
        # # self.conv5_3.weight.data = self.vgg[28].weight.data
        # self.pool5 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # self.relu = torch.nn.ReLU()

        # self.ConvBlock = torch.nn.Sequential(
        #     ConvLayer(3, 64, 3, 1), # conv1_1
        #     torch.nn.ReLU(),
        #     ConvLayer(64, 64, 3, 1), # conv1_2
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(2, 2), # pool1
        #     ConvLayer(64, 128, 3, 1), # conv2_1
        #     torch.nn.ReLU(),
        #     ConvLayer(128, 128, 3, 1), # conv2_2
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(2, 2), # pool2
        #     ConvLayer(128, 256, 3, 1), # conv3_1
        #     torch.nn.ReLU(),
        #     ConvLayer(256, 256, 3, 1), # conv3_2
        #     torch.nn.ReLU(),
        #     ConvLayer(256, 256, 3, 1), # conv3_3
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(2, 2), # pool3
        #     ConvLayer(256, 512, 3, 1), # conv4_1
        #     torch.nn.ReLU(),
        #     ConvLayer(512, 512, 3, 1), # conv4_2
        #     torch.nn.ReLU(),
        #     ConvLayer(512, 512, 3, 1), # conv4_3
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(2, 2), # pool4
        #     ConvLayer(512, 512, 3, 1), # conv5_1
        #     torch.nn.ReLU(),
        #     ConvLayer(512, 512, 3, 1), # conv5_2
        #     torch.nn.ReLU(),
        #     ConvLayer(512, 512, 3, 1), # conv5_3
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(2, 2), # pool5
        # )

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


        # Up-sampling layers
        # self.deconv1 = UpSampleConv(512, 256, kernel_size=3, stride=1, upsample=2)
        # self.deconv2 = UpSampleConv(256, 128, kernel_size=3, stride=1, upsample=2)
        # self.deconv3 = UpSampleConv(128, 64, kernel_size=3, stride=1, upsample=2)
        # self.deconv4 = UpSampleConv(64, 32, kernel_size=3, stride=1, upsample=2)
        # self.deconv5 = UpSampleConv(32, 3, kernel_size=3, stride=1, upsample=2)

        # self.DeConvBlock = torch.nn.Sequential(
        #     UpSampleConv(512, 512, 3, 2, 1), # deconv5_1
        #     torch.nn.ReLU(),
        #     UpSampleConv(512, 256, 3, 2, 1), # deconv4_1
        #     torch.nn.ReLU(),
        #     # UpSampleConv(256, 256, 3, 2, 1), # deconv4_2
        #     # torch.nn.ReLU(),
        #     UpSampleConv(256, 128, 3, 2, 1), # deconv3_1
        #     torch.nn.ReLU(),
        #     UpSampleConv(128, 128, 3, 2, 1), # deconv3_2
        #     torch.nn.ReLU(),
        #     UpSampleConv(128, 64, 3, 2, 1), # deconv2_1
        #     torch.nn.ReLU(),
        #     UpSampleConv(64, 32, 3, 2, 1), # deconv2_2
        #     torch.nn.ReLU(),
        #     UpSampleConv(32, 32, 3, 2, 1), # deconv2_2
        #     torch.nn.ReLU(),
        #     ConvLayer(32, 3, 3, 4, norm = 'None')  
        # )    

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

# class UpSampleConv(torch.nn.Module):
#     '''
#     Inspired from http://distill.pub/2016/deconv-checkerboard/
#     '''
#     def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
#         super(UpSampleConv, self).__init__()
#         # get upsample 
#         self.upsample = upsample
#         # we will be using reflection padding to add zeros around the image
#         self.ref_pad = torch.nn.ReflectionPad2d(kernel_size // 2)
#         # define the convolutional layer
#         self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

#     def forward(self, x):
#         input_x = x # save the input
#         # upsample the input
#         if self.upsample:
#             # interpolate the input using nearest neighbor interpolation
#             input_x = torch.nn.functional.interpolate(input_x, mode='nearest', scale_factor=self.upsample)
#         # get the padded input
#         padded_x = self.ref_pad(input_x)
#         # get the output of the convolutional layer
#         output_x = self.conv(padded_x)
#         # return the output
#         return output_x


# # Program End
    