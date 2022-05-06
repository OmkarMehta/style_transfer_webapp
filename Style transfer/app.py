from io import BytesIO
import base64

import torch
from torchvision import transforms
import cv2
import PIL

from custom_model import CustomModel
from vgg16 import VGG16
import utils

from fast_dash import FastDash, Fastify
from fast_dash.Components import UploadImage, Image, html
from fast_dash.utils import pil_to_b64
from dash import dcc


#### Define inference function
## VGG16 mapper
vgg16_model_mapper = {'coco_rain_princess': 'models/VGG16/COCO/rain_princess.pth',
                         'coco_the_scream':  'models/VGG16/COCO/the_scream.pth',
                         'coco_the_shipwreck':  'models/VGG16/COCO/the_shipwreck.pth',
                         'coco_udnie':  'models/VGG16/COCO/udnie.pth',
                         'coco_wave':  'models/VGG16/COCO/wave.pth',
                         'tinyIN_rain_princess': 'models/VGG16/TinyImagenet/rain_princess.pth',
                         'tinyIN_the_scream':  'models/VGG16/TinyImagenet/the_scream.pth',
                         'tinyIN_the_shipwreck':  'models/VGG16/TinyImagenet/the_shipwreck.pth',
                         'tinyIN_udnie':  'models/VGG16/TinyImagenet/udnie.pth',
                         'tinyIN_wave':  'models/VGG16/TinyImagenet/wave.pth'}


custom_model_mapper = {'coco_rain_princess': 'models/Custom/COCO/rain_princess.pth',
                         'coco_the_scream':  'models/Custom/COCO/the_scream.pth',
                         'coco_the_shipwreck':  'models/Custom/COCO/the_shipwreck.pth',
                         'coco_udnie':  'models/Custom/COCO/udnie.pth',
                         'coco_wave':  'models/Custom/COCO/wave.pth',
                         'tiny_imagenet_rain_princess': 'models/Custom/TinyImagenet/rain_princess.pth',
                         'tiny_imagenet_the_scream':  'models/Custom/TinyImagenet/the_scream.pth',
                         'tiny_imagenet_the_shipwreck':  'models/Custom/TinyImagenet/the_shipwreck.pth',
                         'tiny_imagenet_udnie':  'models/Custom/TinyImagenet/udnie.pth',
                         'tiny_imagenet_wave':  'models/Custom/TinyImagenet/wave.pth'}
                         
def make_snake_case(x): 
    return x.lower().replace(' ', '_')


def load_image_base64(base64_str):
    img = PIL.Image.open(BytesIO(base64.b64decode(base64_str))).convert('RGB')
    return img


def saveimg(image):
    # clip the image to [0, 255]
    image = image.clip(0, 255).astype("uint8")
    image = PIL.Image.fromarray(image)  
    return image


def stylize(image, architecture, trained_on, style):
    
    _, image_content = image.split(',')
    content_image = load_image_base64(image_content)
    
    device = torch.device("cpu")

    if architecture == 'VGG16':
        style_model = VGG16()
        model_path = vgg16_model_mapper[f"{make_snake_case(trained_on)}_{make_snake_case(style)}"]
        
    else:
        style_model = CustomModel()
        model_path = custom_model_mapper[f"{make_snake_case(trained_on)}_{make_snake_case(style)}"]
    
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    
    content_image = content_transform(content_image).unsqueeze(0).to(device)

    with torch.no_grad():
        state_dict = torch.load(model_path)
                
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        
        output = style_model(content_image).cpu()
        output = utils.ttoi(output.clone())
        
    image = saveimg(output)
    image_b64 = pil_to_b64(image)
    
    return image_b64


### Web app using Fast Dash!
# Fastify Dash's dropdown component
architecture_dropdown = Fastify(component=dcc.Dropdown(options={x:x for x in ['VGG16', 'Custom']}), assign_prop='value')
architecture_trained_on = Fastify(component=dcc.Dropdown(options={x:x for x in ['COCO', 'Tiny Imagenet']}), assign_prop='value')
architecture_style = Fastify(component=dcc.Dropdown(options={x:x for x in ['Rain Princess', 'The Scream', 'The Shipwreck',
                                                                                   'Udnie', 'Wave']}), assign_prop='value')

app = FastDash(callback_fn=stylize, 
                inputs=[UploadImage, architecture_dropdown, architecture_trained_on, architecture_style], 
                outputs=Image, 
                title='Neural Style Transfer',
                title_image_path='https://raw.githubusercontent.com/dkedar7/fast_dash/main/examples/Neural%20style%20transfer/assets/icon.png',
                subheader="Apply styles from well-known pieces of art to your own photos",
                github_url='https://github.com/dkedar7/fast_dash/',
                linkedin_url='https://linkedin.com/in/dkedar7/',
                twitter_url='https://twitter.com/dkedar7/',
                theme='JOURNAL')


if __name__=='__main__':
    app.run()