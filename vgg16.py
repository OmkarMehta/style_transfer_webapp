import torch 
from torchvision import models # import the pre-trained model
from collections import namedtuple # import namedtuple for the namedtuple

class VGG16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        # load the pre-trained model and get features
        self.vgg = models.vgg16(pretrained=True).features
        # we are dividing the whole architecture into four parts: each of which has a relu output as the last layer
        self.group1 = torch.nn.Sequential()
        # add the  modules of the first group
        for i in range(4):
            self.group1.add_module(str(i), self.vgg[i])
        self.group2 = torch.nn.Sequential()
        # add the modules of the second group
        for i in range(4, 9):
            self.group2.add_module(str(i), self.vgg[i])
        self.group3 = torch.nn.Sequential()
        # add the modules of the third group
        for i in range(9, 16):
            self.group3.add_module(str(i), self.vgg[i])
        self.group4 = torch.nn.Sequential()
        # add the modules of the fourth group
        for i in range(16, 23):
            self.group4.add_module(str(i), self.vgg[i])
        # if gradients are not required, update parameters to have no gradients
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    def forward(self, x):
        # get the output of the first group
        x1 = self.group1(x)
        # get the output of the second group
        x2 = self.group2(x1)
        # get the output of the third group
        x3 = self.group3(x2)
        # get the output of the fourth group
        x4 = self.group4(x3)
        # create a namedtuple to store the output of the four groups
        out_groups = namedtuple('out_groups', ['x1', 'x2', 'x3', 'x4'])
        # get the output of the four groups
        out = out_groups(x1, x2, x3, x4)
        # return the output of the four groups
        return out



