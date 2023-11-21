import torch
from torch import nn
from peft import LoraConfig
from dataloader import PairedDataset, DataLoader
from encoder import Encoder
from decoder import Decoder

class NST(nn.Module):
    def __init__(self, encoder_num_layers=6, decoder_num_layers=3, lora_config_c=None, lora_config_s=None):
        super().__init__()
        self.encoder_c = Encoder(encoder_num_layers, lora_config_c)
        self.encoder_s = Encoder(encoder_num_layers, lora_config_s)
        self.decoder = Decoder(trans_decoder_layers_num=decoder_num_layers)

    def forward(self, content_image, style_image):
        content, content_pos = self.encoder_c(content_image)
        style, _ = self.encoder_s(style_image)
        output = self.decoder(content, style, content_pos)
        return output
