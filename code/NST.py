import torch
from torch import nn
from peft import LoraConfig
from dataloader import PairedDataset, DataLoader
from encoder import Lora_Encoder
from decoder import Decoder

class NST(nn.Module):
    def __init__(self, lora_config_c, lora_config_s, encoder_num_layers=6, decoder_num_layers=3):
        super().__init__()
        self.encoder_c = Lora_Encoder(encoder_num_layers, lora_config_c)
        self.encoder_s = Lora_Encoder(encoder_num_layers, lora_config_s)
        self.decoder = Decoder(trans_decoder_layers_num=decoder_num_layers)

    def forward(self, content_image, style_image):
        content, content_pos = self.encoder_c(content_image)
        style, _ = self.encoder_s(style_image)
        output = self.decoder(content, style, content_pos)
        return output

if __name__ == '__main__':
    batch_size = 8
    content_dir = "/data/louis/NST_dataset/COCO2014/"
    style_dir = "/data/louis/NST_dataset/WikiArt_processed/"
    train_dataset = PairedDataset(content_dir, style_dir, crop = False, norm = True, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    lora_config_c = lora_config_s = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["query", "value", "key"],
            lora_dropout=0.1,
            bias="none",
        )
    model = NST(lora_config_c, lora_config_s)
    contents, styles = next(iter(train_dataloader))
    # contents = torch.rand(4, 3, 512, 512).abs()
    # styles = torch.rand(4, 3, 512, 512).abs()
    print(contents.shape, styles.shape)
    output = model(contents, styles)
    print(output.shape)
