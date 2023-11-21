import torch
from peft import LoraConfig
from torch.utils.data import DataLoader
import random
import numpy as np
from dataloader import PairedDataset
from model import NST
from lossnet import VGG
from trainer import train_NST
from encoder import print_trainable_parameters

device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')
if __name__ == "__main__":
    # set random seeds
    torch.manual_seed(324)
    torch.cuda.manual_seed(324)
    np.random.seed(324)
    random.seed(324)
    # define dataloader
    print('Loading dataset...')
    content_dir = "/data/louis/NST_dataset/COCO2014/"
    style_dir = "/data/louis/NST_dataset/WikiArt_processed/"
    train_dataset = PairedDataset(content_dir, style_dir, crop = False, norm = True, mode='train')
    val_dataset = PairedDataset(content_dir, style_dir, crop = False, norm = True, mode='val')
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    # define model
    print('Loading model...')
    lora_config_c = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["query", "value", "key"],
        lora_dropout=0.1,
        bias="none",
    )
    lora_config_s = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["query", "value", "key"],
        lora_dropout=0.1,
        bias="none",
    )
    model = NST(encoder_num_layers=6, decoder_num_layers=3)
    # model = NST(encoder_num_layers=6, decoder_num_layers=3, lora_config_c=lora_config_c, lora_config_s=lora_config_s)
    model = model.to(device)
    print_trainable_parameters(model)
    # load the frozen vgg19
    vgg = VGG().to(device)
    print_trainable_parameters(vgg)
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 20, 25], gamma=0.3)
    # train the model
    train_NST(model, vgg, optimizer, lr_scheduler, train_dataloader, val_dataloader, num_epoch=30, comment='1st', save_list=[9, 19, 29], device=device, lambda_c=10, lambda_s=5, lambda_id1=1, lambda_id2=1)



