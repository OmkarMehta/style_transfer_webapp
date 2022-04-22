import torch
from PIL import Image

def load_image(filename, size=None, scale=None):
    img = Image.open(filename).convert('RGB') # convert to RGB
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS) # resize the image
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS) # rescale the image
    return img

def normalize_batch(batch):
    # normalize using imagenet mean and std, obtained from internet
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

def gram_matrix(y):
    # get the batch size, number of channels, and height and width
    (b, ch, h, w) = y.size()
    # reshape the tensor to get the features for each channel
    features = y.view(b, ch, w * h)
    # get the gram matrix
    gram = torch.bmm(features, features.transpose(1, 2))/ (ch * h * w)
    return gram