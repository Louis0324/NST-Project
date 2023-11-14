import torch
from torchsummary import summary
import matplotlib.pyplot as plt
import json
import argparse
import glob
import os
from PIL import Image

from gen_img import *
from loss_fn import *
from model import *
from train import *

def load_json_to_dict(file_path = 'config.json'):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def generate_art(logs_dir, model, config, 
                cont_img_path, style_img_path, 
                style_name_2_index, style_index_2_name,
                cont_name_2_index, cont_index_2_name,
                save_dir_path):
    
    print("cont_img_path: ", cont_img_path)
    print("style_img_path: ", style_img_path)
    
    cont_image = Image.open(cont_img_path).convert("RGB")
    style_image = Image.open(style_img_path).convert("RGB")
    
    width1, height1 = cont_image.size
    width2, height2 = style_image.size
    
    print(f"cont size: ({width1},{height1})")
    print(f"style size: ({width2},{height2})")
    
    width = min(width1, width2)
    height = min(height1, height2)
    
    adjust_crop_size = [config['crop_size'][0], config['crop_size'][1]]
    
    if(config['crop_size'][0] > height):
        adjust_crop_size[0] = height
    if(config['crop_size'][1] > width):
        adjust_crop_size[1] = width
        
    print(f"final crop_size: ({adjust_crop_size[0]},{adjust_crop_size[1]})")
            
    style_outputs, content_outputs = Gen_style_content(
            model=model,
            crop_size=adjust_crop_size,
            content_path=cont_img_path,
            style_path=style_img_path,
            style_name_2_index=style_name_2_index,
            style_index_2_name=style_index_2_name,
            cont_name_2_index=cont_name_2_index,
            cont_index_2_name=cont_index_2_name
        )
            
    gen_img = Init_Gen_Image(
        raw_image_path=cont_img_path,
        crop_size=adjust_crop_size,
        noise_rate=config['noise_rate']
    )
            
    cont_name, _ = os.path.splitext(os.path.basename(cont_img_path))
    style_name, _ = os.path.splitext(os.path.basename(style_img_path))
    name = cont_name + "_instyle_" + style_name
            
    gen_batch = train(
        logs_dir = logs_dir,
        model = model,
        gen_img=gen_img,
        param=config,
        crop_size = adjust_crop_size,
        save_dir = save_dir_path,
        name = name,
        style_outputs=style_outputs,
        content_outputs=content_outputs,
    )


if __name__ == '__main__':
    # Parse command line arguments
    desc = "some hyperparam"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "-o", "--cont_data_dir", required=False, default="cont_images/",
        help="default cont dir_path is ../cont_images/"
    )
    parser.add_argument(
        "-i", "--cont_img_path", required=False, default="cont_images/hongkong.jpg",
        help="default single cont img path is ../cont_images/hongkong.jpg"
    )
    parser.add_argument(
        "-d", "--style_data_dir", required=False, default="style_images/",
        help="default style dir_path is ../style_images/"
    )
    parser.add_argument(
        "-l", "--style_img_path", required=False, default="style_images/guernica.jpg",
        help="default single style img path is ../style_images/guernica.jpg"
    )
    parser.add_argument(
        "-s", "--save_dir", required=False, default="outputs/",
        help="default save_path is ../outputs/"
    )
    parser.add_argument(
        "-m", "--mode", required=False, default="batch_in_all_styles",
        help="single --> process single image in one style once; \
              single_in_all_styles --> process single image in all styles; \
              batch --> process batch images in one style; \
              batch_in_all_styles --> process batch images in all styles"
    )
    parser.add_argument(
        "-c", "--config", required=False, default="config.json",
        help="default config_path is /config.json"
    )
    parser.add_argument(
        "-g", "--logs", required=False, default="logs/",
        help="default path of tensorboard logs is logs/"
    )
    args = parser.parse_args()
    
    cont_dir_path = args.cont_data_dir
    cont_img_path = args.cont_img_path
    style_dir_path = args.style_data_dir
    style_img_path = args.style_img_path
    save_dir_path = args.save_dir
    config_path = args.config
    logs_dir_path = args.logs
    
    directory_name = os.path.basename(os.getcwd())
    if(directory_name == "code"):
        cont_dir_path = "../" + cont_dir_path
        cont_img_path = "../" + cont_img_path
        style_dir_path = "../" + style_dir_path
        style_img_path = "../" + style_img_path
        save_dir_path = "../" + save_dir_path
        config_path = "../" + config_path
        logs_dir_path = "../" + logs_dir_path
    
    with open(config_path, 'r') as json_file:
        # load config as dict
        config = json.load(json_file)
        
    print("config: \n", config)
    style_name_2_index = config['style_name_2_index']
    style_index_2_name = config['style_index_2_name']
    style_weight_dict = config['style_weight_dict']
    cont_name_2_index = config['cont_name_2_index']
    cont_index_2_name = config['cont_index_2_name']
    gen_name_2_index = config['gen_name_2_index']
    gen_index_2_name = config['gen_index_2_name']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load the model
    model = CustomVGG()
    model.to(device)
    
    mode = args.mode
    if(mode == "single"):
        
        generate_art(
            logs_dir=logs_dir_path, model=model, config=config, cont_img_path=cont_img_path,
            style_img_path=style_img_path, style_name_2_index=style_name_2_index,
            style_index_2_name=style_index_2_name, cont_name_2_index=cont_name_2_index,
            cont_index_2_name=cont_index_2_name, save_dir_path=save_dir_path
        )
        
    elif(mode == "single_in_all_styles"):
        style_image_paths = glob.glob(style_dir_path + "*")
        
        for style_img_path in style_image_paths:
            
            generate_art(
                logs_dir=logs_dir_path, model=model, config=config, cont_img_path=cont_img_path,
                style_img_path=style_img_path, style_name_2_index=style_name_2_index,
                style_index_2_name=style_index_2_name, cont_name_2_index=cont_name_2_index,
                cont_index_2_name=cont_index_2_name, save_dir_path=save_dir_path
            )
        
    elif(mode == "batch"):
        cont_image_paths = glob.glob(cont_dir_path + "*")
        
        for cont_img_path in cont_image_paths:
            
            generate_art(
                logs_dir=logs_dir_path, model=model, config=config, cont_img_path=cont_img_path,
                style_img_path=style_img_path, style_name_2_index=style_name_2_index,
                style_index_2_name=style_index_2_name, cont_name_2_index=cont_name_2_index,
                cont_index_2_name=cont_index_2_name, save_dir_path=save_dir_path
            )
            
    elif(mode == "batch_in_all_styles"):
        cont_image_paths = glob.glob(cont_dir_path + "*")
        style_image_paths = glob.glob(style_dir_path + "*")
        
        for cont_img_path in cont_image_paths:
            for style_img_path in style_image_paths:
            
                generate_art(
                    logs_dir=logs_dir_path, model=model, config=config, cont_img_path=cont_img_path,
                    style_img_path=style_img_path, style_name_2_index=style_name_2_index,
                    style_index_2_name=style_index_2_name, cont_name_2_index=cont_name_2_index,
                    cont_index_2_name=cont_index_2_name, save_dir_path=save_dir_path
                )
        
    

    