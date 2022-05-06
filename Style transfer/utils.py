import torch
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets

# def load_image(filename, size=None, scale=None):
#     img = Image.open(filename).convert('RGB') # convert to RGB
#     if size is not None:
#         img = img.resize((size, size), Image.ANTIALIAS) # resize the image
#     elif scale is not None:
#         img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS) # rescale the image
#     return img

# Load image file
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

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. 
    Extends torchvision.datasets.ImageFolder()
    Reference: https://discuss.pytorch.org/t/dataloader-filenames-in-each-batch/4212/2
    """
    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        # the image file path
        path = self.imgs[index][0]

        # make a new tuple that includes original and the path
        tuple_with_path = (*original_tuple, path)
        return tuple_with_path

# def normalize_batch(batch):
#     # normalize using imagenet mean and std, obtained from internet
#     mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
#     std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
#     batch = batch.div_(255.0)
#     return (batch - mean) / std

def gram(y):
    # get the batch size, number of channels, and height and width
    (b, ch, h, w) = y.size()
    # reshape the tensor to get the features for each channel
    features = y.view(b, ch, w * h)
    # get the gram matrix
    gram = torch.bmm(features, features.transpose(1, 2))/ (ch * h * w)
    return gram

# def rescale(x):
#     low, high = x.min(), x.max()
#     x_rescaled = (x - low) / (high - low)
#     return x_rescaled

# def save_image(filename, data):
#     transform = transforms.Compose([
#         transforms.Lambda(lambda x: x.div_(255)),
#         transforms.Normalize(mean=[0, 0, 0], std=[1.0 / s for s in [0.229, 0.224, 0.225]]),
#         transforms.Normalize(mean=[-m for m in [0.485, 0.456, 0.406]], std=[1, 1, 1]),
#         transforms.Lambda(rescale),
#         transforms.ToPILImage(),
#     ])
#     data = transform(data)
#     # convert the data to numpy array
#     img = data.clone().clamp(0, 255).numpy()
#     # reshape the numpy array
#     img = img.transpose(1, 2, 0).astype("uint8")
#     # save the image
#     img = Image.fromarray(img).convert("RGB")
#     img.save(filename)