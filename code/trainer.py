import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.utils import save_image, make_grid
from torchvision.transforms import transforms
from PIL import Image
from lossnet import *

def train_NST(model, vgg, optimizer, lr_scheduler, train_dataloader, val_dataloader, num_epoch, comment, save_list, device, lambda_c, lambda_s, lambda_id1, lambda_id2):
    print(f'Starting training on {device}...')
    writer = SummaryWriter('logs/'+comment)
    val_step = 0
    previous_loss = 10e5
    for epoch in range(num_epoch):
        model.train()
        interval = 20
        val_interval = 400
        # interval = 1
        # val_interval = 10
        running_loss = 0.
        running_c = 0.
        running_s = 0.
        running_id1 = 0.
        running_id2 = 0.
        for i, batch in enumerate(train_dataloader):
            if i % interval == 0:
                start = time.time()
            # load batch data
            contents, styles = batch
            contents = contents.to(device).float()
            styles = styles.to(device).float()
            # run through the model with Ic and Is
            gens = model(contents, styles)
            # run through the model with Ic and Ic (for identity loss)
            Icc = model(contents, contents)
            # run through the model with Is and Is (for identity loss)
            Iss = model(styles, styles)
            # loss is composed of content loss, style loss, identity loss 1, and identity loss 2
            loss, loss_c, loss_s, loss_id1, loss_id2 = calc_total_loss(vgg, gens, contents, Icc, styles, Iss, lambda_c, lambda_s, lambda_id1, lambda_id2)
            running_loss += loss
            running_c += loss_c
            running_s += loss_s
            running_id1 += loss_id1
            running_id2 += loss_id2
            # backprop and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # stats and writer
            if (i+1) % interval == 0:
                end = time.time()
                elapsed_time = end - start
                writer.add_scalar('Total_loss_training', running_loss / interval, epoch*len(train_dataloader)+i)
                writer.add_scalar('Content_loss_training', running_c / interval, epoch*len(train_dataloader)+i)                
                writer.add_scalar('Style_loss_training', running_s / interval, epoch*len(train_dataloader)+i)
                print(f'#################RUNTIME: {elapsed_time}#################')
                print(f'Batch [{i+1}/{len(train_dataloader)}], Total loss: {running_loss/interval}, Content loss: {running_c/interval}, Style loss: {running_s/interval}, id loss 1: {running_id1/interval}, id loss 2: {running_id2/interval}\n')
                running_loss = 0.
                running_c = 0.
                running_s = 0.
                running_id1 = 0.
                running_id2 = 0.
            if (i+1) % val_interval == 0:
                if epoch in save_list and (i+1) == val_interval:
                    save_img = True
                else:
                    save_img = False
                previous_loss = val_NST(model, vgg, writer, val_dataloader, val_step+1, previous_loss, save_img, comment, device, lambda_c, lambda_s)
                val_step += 1
        # one epoch ends
        lr_scheduler.step()
        torch.save(model.state_dict(), os.path.join('/data/louis/NST_checkpoint/', 'NST_'+comment+'_running.pth'))
        
def val_NST(model, vgg, writer, val_dataloader, val_step, previous_loss, save_img, comment, device, lambda_c, lambda_s):
    max_val = 2000
    # max_val = 10
    model.eval()
    with torch.no_grad():
        print(f'Starting validation...')
        running_loss = 0.
        running_c = 0.
        running_s = 0.
        for i, batch in enumerate(val_dataloader):
            # load batch data
            contents, styles = batch
            contents = contents.to(device).float()
            styles = styles.to(device).float()
            # run through the model with Ic and Is
            gens = model(contents, styles)
            # calculate content loss and style loss
            loss_c = content_loss(vgg, contents, gens)
            loss_s = style_loss(vgg, styles, gens)
            running_c += loss_c
            running_s += loss_s
            running_loss += lambda_c*loss_c + lambda_s*loss_s
            if i == max_val:
                break
        # stats and writer
        loss_val = running_loss / (i+1)
        writer.add_scalar('loss_val', loss_val, val_step)
        writer.add_scalar('Content_loss_val', running_c / (i+1), val_step)                
        writer.add_scalar('Style_loss_val', running_s / (i+1), val_step)
        print(f'loss_val: {loss_val}, Content loss: {running_c/(i+1)}, Style loss: {running_s/(i+1)}\n')
        # visualization and save
        img_grid = val_selected_images(model, device, writer, val_step, paths_c, paths_s, transform)
        if save_img:
            print(f'saving images...')
            save_image(img_grid, f'/data/louis/saved_gen/{comment}_Images_val_{val_step}.png')
        # save better model
        if previous_loss > loss_val:
            print(f'saving better model...')
            torch.save(model.state_dict(), os.path.join('/data/louis/NST_checkpoint/', 'NST_'+comment+'_best.pth'))
    return min(previous_loss, loss_val)

transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

paths_c = [
    '/data/louis/NST_dataset/COCO2014/val/COCO_val2014_000000000073.jpg',
    '/data/louis/NST_dataset/COCO2014/val/COCO_val2014_000000000359.jpg',
    '/data/louis/NST_dataset/COCO2014/val/COCO_val2014_000000213412.jpg',
    '/data/louis/NST_dataset/COCO2014/val/COCO_val2014_000000352491.jpg'
]

paths_s = [
    '/data/louis/NST_dataset/WikiArt_processed/val/albrecht-altdorfer_beheading-of-st-catherine-1506.jpg',
    '/data/louis/NST_dataset/WikiArt_processed/val/ad-reinhardt_collage-1940.jpg', 
    '/data/louis/NST_dataset/WikiArt_processed/val/albrecht-altdorfer_alpine-landscape-with-church-1522.jpg',
    '/data/louis/NST_dataset/WikiArt_processed/val/a.y.-jackson_algoma-in-november-1935.jpg'
]

def val_selected_images(model, device, writer, val_step, paths_c, paths_s, transform):
    contents = []
    styles = []
    num_selected = len(paths_c)
    for file_c, file_s in zip(paths_c, paths_s):
        content_imag = Image.open(file_c).convert("RGB")
        Ic = transform(content_imag)
        contents.append(Ic)
        style_imag = Image.open(file_s).convert("RGB")
        Is = transform(style_imag)
        styles.append(Is)
    contents = torch.stack(contents, dim=0).to(device)
    styles = torch.stack(styles, dim=0).to(device)
    gens = model(contents, styles)
    img_stack = torch.vstack([contents, styles, gens])
    img_grid = make_grid(img_stack, normalize=True, nrow=num_selected)
    writer.add_image(
        "Contents/Styles/Generated", img_grid, global_step=val_step
    )
    return img_grid
    
        
