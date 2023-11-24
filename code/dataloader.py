import os
import random
from tqdm import tqdm
import numpy as np
import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from matplotlib import pyplot as plt

class PairedDataset(Dataset):
    def __init__(self, content_dir, style_dir, crop = False, norm = True, mode='train'):
        
        """_summary_

        param content_dir: dir path of content images
        param style_dir: dir path of style images
        param crop: whether to crop the image (in (224,224) patch)
        !! if crop == False, then we resize the image into (224,224) !!
        
        param norm: whether to normalize the image
        
        param mode: "train" or "val
        
        """
        
        if(mode == "train"):
            self.content_dir = os.path.join(content_dir, "train")
            self.style_dir = os.path.join(style_dir, "train")
        elif(mode == "val"):
            self.content_dir = os.path.join(content_dir, "val")
            self.style_dir = os.path.join(style_dir, "val")
        else:
            raise ValueError("Dataset mode should be 'train' or 'val'")
        
        self.mode = mode
        
        self.transform = self._build_transform(crop, norm)

        self.content_files = self._get_image_files(self.content_dir)
        self.style_files = self._get_image_files(self.style_dir)
            
    def _build_transform(self, crop, norm):
        transform_list = []

        if crop:
            transform_list.append(transforms.RandomCrop(224, 224))
        else:
            transform_list.append(transforms.Resize((224,224)))
            
        transform_list.append(transforms.ToTensor())

        if norm:
            transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        return transforms.Compose(transform_list)

    def __len__(self):
        # As the number of content images is higher than that of style images
        
        return len(self.style_files)

    def __getitem__(self, index):
        content_file = self.content_files[index]
        style_file = self.style_files[index]

        content_image = Image.open(content_file).convert("RGB")
        
        # some content images's size are smaller than (224,224)
        # So we need to resize them to (224, 224)
        width, height = content_image.size
        if width < 224 or height < 224:
            content_image = content_image.resize((224, 224), Image.LANCZOS)
        
        style_image = Image.open(style_file).convert("RGB")

        content_image = self.transform(content_image)
        style_image = self.transform(style_image)

        return content_image, style_image

    def _get_image_files(self, directory):
        image_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith((".png", ".jpg", ".jpeg")):
                    image_files.append(os.path.join(root, file))
        return image_files
    
    
# simply test this module
    
if __name__ == "__main__":
    
    batch_size = 4
    content_dir = "G:/COCO2014/"
    style_dir = 'G:/WikiArt_processed/' 

    # build train dataet and loader
    train_dataset = PairedDataset(content_dir, style_dir, crop = False, norm = False, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # build val dataet and loader
    val_dataset = PairedDataset(content_dir, style_dir, crop = False, norm = False, mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # test train dataset and loader
    print("test train dataset and loader:")
    train_index = 0
    for content_image, style_image in train_dataloader:
        if(train_index == 0):
            print("content_image shape: ", content_image.shape)
            print("style_image shape: ", style_image.shape)
            break
        train_index += 1
    #print("train: f{train_index} iterations per epoch (batch_size={batch_size})\n")

    print("test val dataset and loader:")
    val_index = 0
    for content_image, style_image in val_dataloader:
        if(val_index == 0):
            print("content_image shape: ", content_image.shape)
            print("style_image shape: ", style_image.shape)
            break
        train_index += 1
    #print("val: f{train_index} iterations per epoch (batch_size={batch_size})")
    
