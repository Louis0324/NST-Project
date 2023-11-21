import torch
from transformers import ViTImageProcessor, ViTModel, ViTConfig
import copy
from torch import nn
from peft import LoraConfig, get_peft_model

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

def ModifyLayers(model, num_layers):
    """select the first num_layers encoder layers, and change the last pooler into identity
    
    """
    # select the first num_layers encoder layers
    encoder = model.encoder.layer
    model.encoder.layer = encoder[:num_layers]
    # delete the last pooler layer
    model.pooler = Identity()
    return model

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

class Encoder(nn.Module):
    def __init__(self, num_layers, lora_config=None):
        super().__init__()
        model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        model = ModifyLayers(model, num_layers)
        if lora_config is not None:
            self.model = get_peft_model(model, peft_config=lora_config)
        else:
            self.model = model
        
    def forward(self, imags):
        B, C, H, W = imags.shape
        if H == 224 and W == 224: # training config
            interpolate_pos_encoding = False
        else: # test time config
            interpolate_pos_encoding = True
        outputs = self.model(imags, interpolate_pos_encoding=interpolate_pos_encoding)
        out = outputs.last_hidden_state[:, 1:, :]
        pe = self.model.embeddings.interpolate_pos_encoding(out, H, W)[:, 1:, :]
        return out, pe

if __name__ == '__main__':
    lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["query", "value", "key"],
            lora_dropout=0.1,
            bias="none",
        )
    lora_encoder = Encoder(num_layers=6)
    # lora_encoder = Encoder(num_layers=6, lora_config=lora_config)
    print(lora_encoder)
    print_trainable_parameters(lora_encoder)
    
    img = torch.rand(4, 3, 512, 512).abs()
    out, pe = lora_encoder(img)
    print(out.shape, pe.shape)
