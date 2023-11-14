import torch
from PIL import Image
import numpy as np

from model import Get_preprocess
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Init_Gen_Image(raw_image_path, crop_size, noise_rate = 0.6):
    raw_image = Image.open(raw_image_path).convert("RGB")
    # for PIL-Image, (width, height)
    crop_size = (crop_size[1], crop_size[0])
    raw_image = raw_image.resize(crop_size)

    # trans into np_array
    raw_image_array = np.array(raw_image)

    # create noise image in same shape
    noise_image = np.random.randint(0, 256, size=raw_image_array.shape, dtype=np.uint8)

    # blend raw_image and noise image according to noise rate
    blended_image = np.multiply(raw_image_array, noise_rate) + np.multiply(noise_image, (1-noise_rate))
    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)

    # trans back into PIL image
    result_image = Image.fromarray(blended_image)
    
    return result_image

def Load_img(path):
    return Image.open(path).convert("RGB")
    
def Get_input(content_path, style_path):
    content_img = Load_img(content_path)
    style_img = Load_img(style_path)
    return content_img, style_img

def Gen_style_content(model, crop_size, content_path, style_path, style_name_2_index, style_index_2_name,
                      cont_name_2_index, cont_index_2_name):
    
    content_img, style_img = Get_input(content_path, style_path)
    preprocess = Get_preprocess(crop_size)
    
    content_tensor = preprocess(content_img)
    content_batch = content_tensor.unsqueeze(0).to(device)
    style_tensor = preprocess(style_img)
    style_batch = style_tensor.unsqueeze(0).to(device)
    
    print("content_batch shape: ", content_batch.shape)
    print("style_batch shape: ", style_batch.shape)
    
    style_outputs = {}
    x = style_batch
    for name, module in model.features.named_children():
        x = module(x)
        if name in style_name_2_index.values():
            str_name = style_index_2_name[name]
            style_outputs[str_name] = x
                
    content_outputs = {}
    x = content_batch
    for name, module in model.features.named_children():
        x = module(x)
        if name in cont_name_2_index.values():
            str_name = cont_index_2_name[name]
            content_outputs[str_name] = x
      
    return style_outputs, content_outputs

def Gen_art(model, Gen_batch, gen_name_2_index, gen_index_2_name):
    gen_outputs = {}
    x = Gen_batch
    
    for name, module in model.features.named_children():
        x = module(x)
        if name in gen_name_2_index.values():
            str_name = gen_index_2_name[name]
            gen_outputs[str_name] = x
            
    return gen_outputs