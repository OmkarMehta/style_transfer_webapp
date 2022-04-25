# VGG16 as feature extractor

import torch 
from torchvision import models # import the pre-trained model
from collections import namedtuple # import namedtuple for the namedtuple

# class VGG16(torch.nn.Module):
#     def __init__(self, requires_grad=False):
#         super(VGG16, self).__init__()
#         # load the pre-trained model and get features
#         self.vgg = models.vgg16(pretrained=True).features
#         # we are dividing the whole architecture into four parts: each of which has a relu output as the last layer
#         self.group1 = torch.nn.Sequential()
#         # add the  modules of the first group
#         for i in range(4):
#             self.group1.add_module(str(i), self.vgg[i])
#         self.group2 = torch.nn.Sequential()
#         # add the modules of the second group
#         for i in range(4, 9):
#             self.group2.add_module(str(i), self.vgg[i])
#         self.group3 = torch.nn.Sequential()
#         # add the modules of the third group
#         for i in range(9, 16):
#             self.group3.add_module(str(i), self.vgg[i])
#         self.group4 = torch.nn.Sequential()
#         # add the modules of the fourth group
#         for i in range(16, 23):
#             self.group4.add_module(str(i), self.vgg[i])
#         # if gradients are not required, update parameters to have no gradients
#         if not requires_grad:
#             for param in self.parameters():
#                 param.requires_grad = False
#     def forward(self, x):
#         # get the output of the first group
#         x1 = self.group1(x)
#         # get the output of the second group
#         x2 = self.group2(x1)
#         # get the output of the third group
#         x3 = self.group3(x2)
#         # get the output of the fourth group
#         x4 = self.group4(x3)
#         # create a namedtuple to store the output of the four groups
#         out_groups = namedtuple('out_groups', ['x1', 'x2', 'x3', 'x4'])
#         # get the output of the four groups
#         out = out_groups(x1, x2, x3, x4)
#         # return the output of the four groups
#         return out

class VGG16(torch.nn.Module):
    def __init__(self, vgg_path="models/vgg16-00b39a1b.pth"):
        super(VGG16, self).__init__()
        # load the model of vgg16 with pretrained set to False
        vgg16_features = models.vgg16(pretrained=False)
        # load the weights of vgg16 from umich, cited in the report
        vgg16_features.load_state_dict(torch.load(vgg_path), strict=False)
        # get the features of vgg16
        self.features = vgg16_features.features

        # we don't want to train the model, so we set requires_grad to False because this is a feature extractor
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        layers = {'3': 'relu1_2', '8': 'relu2_2', '15': 'relu3_3', '22': 'relu4_3'}
        # create a dictinary to store the output of each layer
        features = {}
        for idx, layer in self.features._modules.items():
            # forward pass
            x = layer(x)
            if idx in layers:
                features[layers[idx]] = x
                if (idx=='22'):
                    break
        # return the features
        return features
