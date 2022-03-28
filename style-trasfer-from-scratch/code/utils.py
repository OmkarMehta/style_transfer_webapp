import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import PIL 
import matplotlib.pyplot as plt
import numpy as np

def grammatrix(input_features, normalize=False):
    '''
    From the paper:
    "The Gram matrix of a matrix A is the matrix A^T A.
    In other words, the Gram matrix of a matrix A is the matrix of dot products of its columns.

    :Input:
    - input_features: (batch_size, channels, height, width) PyTorch tensor
    - normalize: bool, whether to normalize the Gram matrix
    '''
    # get (batch_size, channels, height, width)
    batch_size, channels, height, width = input_features.size()

    # reshape to (batch_size, channels, height * width)
    features = input_features.view(batch_size, channels, height * width)

    # compute Gram matrix
    # if we normalize the Gram matrix
    if normalize==True:
        gram = torch.bmm(features, features.transpose(1, 2)) / (channels * height * width)
    else:
        gram = torch.bmm(features, features.transpose(1, 2))
    
    # return Gram matrix
    return gram
