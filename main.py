#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import torch

from PIL import Image
from torchvision import transforms
from lib.tool import ResNetGenerator

st.title("An app that turns a horse into zebra using deep learning")
st.info("GAN: Generative Adversarial Network")


#---------------------------------------------------#

# Load an input image for testing

img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if img_file_buffer is not None:
    img = (Image.open(img_file_buffer))


else:
    demo_image = "data/horse.jpeg"
    img = (Image.open(demo_image))


# define a preprocessor for the horse2zebra model
preprocess = transforms.Compose([transforms.Resize(256),
                                 transforms.ToTensor()])

# Preprocess the input image
img_t = preprocess(img)

# Prepare the barch of image to pass to the inference
batch_t = torch.unsqueeze(img_t, 0)

#---------------------------------------------------#

# Load cycleGAN architecture (initialized with random parameter)
netG = ResNetGenerator()

# Download pretrained model on dataset mng.bz/8pKP
model_path = 'model/horse2zebra_0.4.0.pth' # pickle file
model_data = torch.load(model_path) # model's weight

# Load the pretrained horse2zebra weights to the cycleGAN model
netG.load_state_dict(model_data)
netG.eval()

#---------------------------------------------------#

# Inference
batch_out = netG(batch_t)

out_t = (batch_out.data.squeeze() + 1.0) / 2.0

# transform the output image to PIL image object
out_img = transforms.ToPILImage()(out_t)
# out_img.save('../data/p1ch2/zebra.jpg')

#---------------------------------------------------#


st.markdown("## Input")
st.image(img, use_column_width=True)

st.markdown("## Output")
st.image(out_img, use_column_width=True)
