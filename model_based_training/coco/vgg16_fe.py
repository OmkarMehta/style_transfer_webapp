# VGG16 as feature extractor

import torch 
from torchvision import models # import the pre-trained model
from collections import namedtuple # import namedtuple for the namedtuple


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
