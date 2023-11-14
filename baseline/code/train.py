import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
import os

from model import Get_preprocess
from loss_fn import CustomLoss
from gen_img import Gen_art
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_mid_result(gen_batch, iter, save_dir):
    gen_batch = gen_batch.cpu()
    gen_tensor = gen_batch.squeeze(0)
    gen_tensor = gen_tensor.permute(1,2,0)
    gen_image = gen_tensor.detach().numpy()
    
    gen_image = cv2.cvtColor(gen_image, cv2.COLOR_RGB2BGR)
    gen_image = np.clip(gen_image, 0, 1) * 255
    gen_image = gen_image.astype(np.uint8)
    cv2.imwrite(save_dir + str(iter) + ".jpg", gen_image)

def train(logs_dir, model, gen_img, param, crop_size, save_dir, name, style_outputs, content_outputs):
    lr = param['lr']
    iterations = param['iter']
    alpha = param['alpha']
    beta = param['beta']
    save_feq = param['save_feq']
    log_feq = param['log_feq']
    nr = param['noise_rate']
    
    style_name_2_index = param["style_name_2_index"]
    style_index_2_name = param["style_index_2_name"]
    style_weight_dict = param["style_weight_dict"]
    cont_name_2_index = param["cont_name_2_index"]
    cont_index_2_name = param["cont_index_2_name"]
    gen_name_2_index = param["gen_name_2_index"]
    gen_index_2_name = param["gen_index_2_name"]
    
    folder_name = name + "_" + str(nr) + "_" + str(lr) + "_" + str(alpha) + "_" + \
                    str(beta) + "_(" + str(crop_size[0]) + "," + str(crop_size[1]) + ")_"
    if(style_weight_dict["conv1_1"] == 0.2):
        folder_name += "average/"
    else:
        folder_name += "style/"
        
    folder_path = save_dir + folder_name
    
    writer_path = logs_dir + "/" + folder_name
    writer = SummaryWriter(writer_path)
    print(f"log will be written to {writer_path}")
        
    if not os.path.exists(folder_path):
        # mkdir
        os.mkdir(folder_path)
        print(f"mkdir save_folder {folder_path}")
    
    preprocess = Get_preprocess(crop_size)
    gen_tensor = preprocess(gen_img)
    gen_batch = gen_tensor.unsqueeze(0).to(device)
    print("gen_batch shape: ", gen_batch.shape)
    gen_batch.requires_grad = True
    
    optimizer = optim.Adam([gen_batch], lr = lr)
    loss_fn = CustomLoss(style_name_2_index, style_weight_dict, cont_name_2_index, alpha, beta)
    
    for i in range(iterations):
    
        gen_outputs = Gen_art(model, gen_batch, gen_name_2_index, gen_index_2_name)
        loss = loss_fn(gen_outputs, style_outputs, content_outputs)
        
        writer.add_scalar("Generate Loss", loss, i)
        if(i % log_feq == 0):
            print(f"iter {i} loss :{loss}")
        
        # backward and update
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        
        if(i % save_feq == 0):
            save_mid_result(gen_batch, i, folder_path)
        
    writer.close()
    return gen_batch