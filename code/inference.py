import torch
from torch import nn
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model
from torchvision.transforms import transforms
from PIL import Image
from model import NST

def generate_image(model, content_path, style_path, device):
    # load images
    content_imag = transforms.ToTensor()(Image.open(content_path).convert("RGB"))
    style_imag = transforms.ToTensor()(Image.open(style_path).convert("RGB"))
    # calculate input standard size, since the input should be square images and size should be a multiple of 16
    _, Hc, Wc = content_imag.shape
    style_imag_resize = transforms.Resize((Hc, Wc))(style_imag)
    size = max(Hc, Wc)
    if size % 16 != 0:
        std_size = size + (16 - size % 16)
    else:
        std_size = size
    pad_H = std_size - Hc
    pad_W = std_size - Wc
    # pad the images to be square input
    Ic = nn.ReflectionPad2d([0, pad_W, 0, pad_H])(content_imag).unsqueeze(0).to(device)
    Is = nn.ReflectionPad2d([0, pad_W, 0, pad_H])(style_imag_resize).unsqueeze(0).to(device)
    # resize Ic and Is if std_size too big
    max_size = 512
    if std_size > max_size:
        Ic = transforms.Resize((max_size, max_size))(Ic)
        Is = transforms.Resize((max_size, max_size))(Is)
    # normalize
    normalize = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    Ic = normalize(Ic)
    Is = normalize(Is)
    # generate images
    model.eval()
    with torch.no_grad():
        Ig = model(Ic, Is)
    # inverse the normalization
    inv_normalize = transforms.Compose([ 
        transforms.Normalize(mean = [0., 0., 0.], std = [1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean = [-0.485, -0.456, -0.406], std = [1., 1., 1.]),
    ])
    Ig = inv_normalize(Ig).cpu().squeeze(0)
    # resize Ig back if Ic and Is being resized before
    if std_size > max_size:
        Ig = transforms.Resize((std_size, std_size))(Ig)
    # crop the generated images to be of the same size as content_imag
    if pad_H == 0 and pad_W == 0:
        gen_imag = Ig
    elif pad_H == 0 and pad_W != 0:
        gen_imag = Ig[:,:,:-pad_W]
    elif pad_H != 0 and pad_W == 0:
        gen_imag = Ig[:,:-pad_H]
    else:
        gen_imag = Ig[:,:-pad_H, :-pad_W]
    # visualize
    plt.figure(figsize=(20, 20))
    plt.subplot(1,3,1)
    plt.imshow(content_imag.permute(1,2,0))
    plt.subplot(1,3,2)
    plt.imshow(style_imag.permute(1,2,0))
    plt.subplot(1,3,3)
    plt.imshow(gen_imag.permute(1,2,0))
    return gen_imag, content_imag, style_imag

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    # config of the best model, please don't change any of these! 
    r = 16
    target_modules = ["query", "value", "key", "dense"]
    encoder_num_layers = 6
    decoder_num_layers = 3
    model_id = '34th'
    model_path = '/data/louis/NST_checkpoint/NST_' + model_id + '_jointly_training_best.pth'

    # lora config
    lora_config_c = LoraConfig(
        r=r,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
    )
    lora_config_s = LoraConfig(
        r=r,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
    )
    # load the model
    model = NST(encoder_num_layers=encoder_num_layers, decoder_num_layers=decoder_num_layers, freeze=True)
    model.encoder_c = get_peft_model(model.encoder_c, lora_config_c)
    model.encoder_s = get_peft_model(model.encoder_s, lora_config_s)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
