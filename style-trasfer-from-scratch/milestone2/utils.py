import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import PIL 
import matplotlib.pyplot as plt
import numpy as np
import cv2

def gram(input_features):
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
    gram = torch.bmm(features, features.transpose(1, 2)) / (channels * height * width)
    # return Gram matrix
    return gram

def loss_of_style(generated_features, MSELoss, style_gram, style_weight):
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
    # # get the number of style layers
    # num_style_layers = len(style_layer_indices)

    # initialize style loss to 0
    style_loss = 0

    # loop over style layers
    for key, value in generated_features.items():
        loss = MSELoss(utils.gram_matrix(value), style_gram[key][:curr_batch_size])
        style_loss += loss
    style_loss *= style_weight
        # # get the index of the style layer
        # style_layer_index = style_layer_indices[i]
        # # get the weight of the style layer
        # weight = weights_of_style_layers[i]
        # # get the Gram matrix of the style layer at the current style layer index
        # style_gram_matrix = grammatrix(input_features[style_layer_index], normalize=True)
        # # compute the style loss
        # style_loss += weight * ((style_gram_matrix - style_gram_matrices[i]) ** 2).sum()
    
    # return style loss
    return style_loss

def loss_of_content(generated_features, content_features, MSELoss, content_weight):
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
    content_loss = content_weight * MSELoss(generated_features['relu2_2'], content_features['relu2_2'])
    return content_loss

def content_transform(content, image_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x[None])
    ])
    return transform(content)

def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled


def detransform_content(content, image_size):
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x[0]),
        transforms.Normalize(mean=[0, 0, 0], std=[1.0 / s for s in [0.229, 0.224, 0.225]]),
        transforms.Normalize(mean=[-m for m in [0.485, 0.456, 0.406]], std=[1, 1, 1]),
        transforms.Lambda(rescale),
        transforms.ToPILImage(),
    ])
    return transform(content)

def load_image(path):
    # Images loaded as BGR
    image = cv2.imread(path)
    return image

# Show image
def show(image):
    # Convert from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # imshow() only accepts float [0,1] or int [0,255] so we will convert the image to float [0,1] and clip it.
    image = np.array(image/255).clip(0,1)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(image)
    plt.show()

# save image
def saveimg(image, image_path):
    # clip the image to [0, 255]
    image = image.clip(0, 255)
    # write the image to file
    cv2.imwrite(image_path, image)

def itot(img, max_size=None):
    # Rescale the image
    if (max_size==None):
        itot_t = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])    
    else:
        H, W, C = img.shape
        image_size = tuple([int((float(max_size) / max([H,W]))*x) for x in [H, W]])
        itot_t = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])

    # Convert image to tensor
    tensor = itot_t(img)

    # Add the batch_size dimension
    tensor = tensor.unsqueeze(dim=0)
    return tensor
def ttoi(tensor):
    # Add the means
    #ttoi_t = transforms.Compose([
    #    transforms.Normalize([-103.939, -116.779, -123.68],[1,1,1])])

    # Remove the batch_size dimension
    tensor = tensor.squeeze()
    #img = ttoi_t(tensor)
    img = tensor.cpu().numpy()
    
    # Transpose from [C, H, W] -> [H, W, C]
    img = img.transpose(1, 2, 0)
    return img

def transfer_color(src, dest):
    """
    Transfer Color using YIQ colorspace. Useful in preserving colors in style transfer.
    This method assumes inputs of shape [Height, Width, Channel] in BGR Color Space
    """
    src, dest = src.clip(0,255), dest.clip(0,255)
        
    # Resize src to dest's size
    H,W,_ = src.shape 
    dest = cv2.resize(dest, dsize=(W, H), interpolation=cv2.INTER_CUBIC)
    
    dest_gray = cv2.cvtColor(dest, cv2.COLOR_BGR2GRAY) #1 Extract the Destination's luminance
    src_yiq = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)   #2 Convert the Source from BGR to YIQ/YCbCr
    src_yiq[...,0] = dest_gray                         #3 Combine Destination's luminance and Source's IQ/CbCr
    
    return cv2.cvtColor(src_yiq, cv2.COLOR_YCrCb2BGR).clip(0,255)  #4 Convert new image from YIQ back to BGR


def plot_loss_hist(c_loss, s_loss, total_loss, title="Loss History", xlabel='Every 500 iterations'):
    x = [i for i in range(len(total_loss))]
    plt.figure(figsize=[10, 6])
    plt.plot(x, c_loss, label="Content Loss")
    plt.plot(x, s_loss, label="Style Loss")
    plt.plot(x, total_loss, label="Total Loss")
    
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel('Loss')
    plt.title(title)
    plt.show()
