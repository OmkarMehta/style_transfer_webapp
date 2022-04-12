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

    :Output:
    - gram_matrix: (channels, channels) PyTorch tensor
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

def loss_of_style(input_features, style_layer_indices, style_gram_matrices, weights_of_style_layers):
    '''
    From the paper:
    "The style loss is a weighted sum of style losses for each style layer.
    The style losses are computed as a weighted sum of squared differences between the Gram matrices of the style features 
    and the Gram matrices of the style features of the style image.

    :Input:
    - input_features: (batch_size, channels, height, width) PyTorch tensor –– it's a list of features of the user's image
    - style_layer_indices: list of ints, indices of style layers –– it's a list of indices to get features given by the style_layers
    - style_gram_matrices: list of PyTorch tensors, Gram matrices of the style image –– it's a list of Gram matrices of the style image, 
                            where style_gram_matrices[i] is the Gram matrix of the style_layer_indices[i]
    - weights_of_style_layers: list of floats, weights of style layers –– it's a list of weights of the style layers, where 
                                weights_of_style_layers[i] is the weight of the style_layer_indices[i]

    :Output:
    - style_loss: float, style loss
    '''
    # get the number of style layers
    num_style_layers = len(style_layer_indices)

    # initialize style loss to 0
    style_loss = 0

    # loop over style layers
    for i in range(num_style_layers):
        # get the index of the style layer
        style_layer_index = style_layer_indices[i]
        # get the weight of the style layer
        weight = weights_of_style_layers[i]
        # get the Gram matrix of the style layer at the current style layer index
        style_gram_matrix = grammatrix(input_features[style_layer_index], normalize=True)
        # compute the style loss
        style_loss += weight * ((style_gram_matrix - style_gram_matrices[i]) ** 2).sum()
    
    # return style loss
    return style_loss

def loss_of_content(input_features, weight_of_content_layer, content_image_features):
    '''
    From the paper:
    The content loss is a weighted sum of squared differences between the features of the content image and the features of the generated image.

    :Input:
    - input_features: (batch_size, channels, height, width) PyTorch tensor –– it's a list of features of the user's image
    - weight_of_content_layer: float, weight of the content layer
    - content_image_features: (batch_size, channels, height, width) PyTorch tensor –– it's a list of features of the content image

    :Output:
    - content_loss: float, content loss

    '''
    # compute the content loss
    content_loss = ((input_features - content_image_features) ** 2).sum() * weight_of_content_layer
    # return content loss
    return content_loss

def content_transform(content, image_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[None])
    ])
    return transform(content)

def detransform_content(content, image_size):
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x[0]),
        transforms.Lambda(rescale),
        transforms.ToPILImage(),
    ])
    return transform(content)