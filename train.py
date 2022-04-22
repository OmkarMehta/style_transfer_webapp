import numpy as np
import torch 
from torchvision import datasets
from torchvision import transforms
from torch.optim import Adam
from vgg16 import VGG16 # this is used to train style transfer
from vgg16_fe import VGG16 as VGG16_FE # this is used to extract features
from custom_model import CustomModel
from torch.utils.data import DataLoader
import utils 

model_zoo = {
    'vgg': VGG16,
    'vgg_fe': VGG16_FE,
    'custom': CustomModel
}

def train(content_weight=1e5, style_weight=1e10, num_epochs=10, batch_size=32, model_name='vgg', style_image_path = 'style/starry_night.jpg'):
    # initialize the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # make the transforms
    transform = transforms.Compose([
        transforms.Resize(128), # resize the image to 256x256
        transforms.CenterCrop(128), # center crop the image to 256x256
        transforms.ToTensor(), # convert the image to a tensor
        transforms.Lambda(lambda x: x.mul(255)) # scale the image.
    ])

    # load the training dataset
    train_dataset = datasets.ImageFolder(root='tiny-224/train', transform=transform)
    # loader the training dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # load the model
    if model_name == 'custom':
        model = model_zoo[model_name]().to(device)
    else:
        model = VGG16(requires_grad=True).to(device)
    
    # our feature extractor is VGG16
    vgg16_fe = model_zoo['vgg_fe']().to(device) # requires_grad is False by default

    # optimizer
    optimizer = Adam(model.parameters(), lr=0.001)
    # loss function
    criterion = torch.nn.MSELoss()

    # style transform
    style_transform = transforms.Compose([
        transforms.ToTensor(), # convert the image to a tensor
        transforms.Lambda(lambda x: x.mul(255)) # scale the image.
    ])

    # load the style image
    style = utils.load_image(style_image_path)
    print(style)
    
    # transform the style image
    style = style_transform(style).to(device)
    print(style.shape)
    # print('Style image is on device: {}'.format(style.device))
    # repeat the style image for each batch
    style = style.repeat(batch_size, 1, 1, 1)
    
    # print the device of the style image
    
    # cuda.FloatTensor 
    style = torch.cuda.FloatTensor(style)
    # print the tensor type of the style image
    print('Style image is of type: {}'.format(style.type()))

    # extract the features of the style image
    features_style = vgg16_fe(utils.normalize_batch(style).to(device))

    # get the gram matrix of the style image
    gram_style = [utils.gram_matrix(y) for y in features_style]

    for epoch in range(num_epochs):
        # train the model
        model.train()
        # initialize the content loss and style loss
        content_loss_total = 0.0
        style_loss_total = 0.0
        # count the number of batches
        batches_done = 0
        for batch_id, (content_images, _) in enumerate(train_loader):
            # get the size of the batch
            n_batch = len(content_images)
            # add this to the total number of batches
            batches_done += n_batch

            # initialize the optimizer
            optimizer.zero_grad()

            # add content_images to device
            content_images = content_images.to(device)
            # pass content images through the model (either vgg or custom)
            generated_images = model(content_images)
            # print the shape of the generated images
            print('Generated images shape: {}'.format(generated_images.shape))

            # normalize the generated image as well as the content images
            # this is done to make the loss function more robust
            generated_images = utils.normalize_batch(generated_images)
            # print the shape of the normalized generated images
            print('Normalized generated images shape: {}'.format(generated_images.shape))
            content_images = utils.normalize_batch(content_images)
            # print the shape of the normalized content images
            print('Normalized content images shape: {}'.format(content_images.shape))

            # extract the features of the generated images and the content images
            features_generated = vgg16_fe(generated_images)
            # print the shape of the features
            print('Features generated shape: {}'.format(features_generated.x2.shape))
            features_content = vgg16_fe(content_images)
            # print the shape of the features
            print('Features content shape: {}'.format(features_content.x2.shape))

            # calculate the content loss
            content_loss = content_weight * criterion(features_generated.x2, features_content.x2)

            # calculate the style loss
            style_loss = 0.0
            for feat_y, gram_st in zip(features_generated, gram_style):
                gram_y = utils.gram_matrix(feat_y) # get the gram matrix of the generated image
                # calculate the style loss between gram matrix of generated image and gram matrix of style image
                style_loss += criterion(gram_y, gram_st[:n_batch, :, :]) 
            style_loss *= style_weight

            # calculate the total loss
            loss = content_loss + style_loss
            # backpropagate the loss
            torch.autograd.set_detect_anomaly(True)
            loss.backward(retain_graph=True)
            # loss.backward(retain_graph=True)
            # update the weights
            optimizer.step()

            # add it to the total content loss and style loss
            content_loss_total += content_loss.item()
            style_loss_total += style_loss.item()

            # print the loss
            if batch_id % 500 == 0:
                print('Epoch {}/{}'.format(epoch + 1, num_epochs))
                print('Content Loss: {:4f}'.format(content_loss_total / (batch_id + 1)))
                print('Style Loss: {:4f}'.format(style_loss_total / (batch_id + 1)))
                print('Total Loss: {:4f}'.format(content_loss_total / (batch_id + 1) + style_loss_total / (batch_id + 1)))
                print()
            
    # save the model
    model.eval().cpu()
    model_filename = '{}_{}.pth'.format(style_image_path.split('/')[-1].split('.')[0], model_name)
    model_path = 'models/{}'.format(model_filename)
    torch.save(model.state_dict(), model_path)

    print(f"Model saved to {model_path}")



    


            






