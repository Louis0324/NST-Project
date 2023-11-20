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

class Lora_Encoder(nn.Module):
    def __init__(self, num_layers, lora_config):
        super().__init__()
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        model = ModifyLayers(model, num_layers)
        self.lora_model = get_peft_model(model, peft_config=lora_config)
        
    def forward(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.lora_model(**inputs)
        return outputs.last_hidden_state[:, 1:, :], self.model.embeddings.position_embeddings[:, 1:, :]

if __name__ == '__main__':
    lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["query", "value", "key"],
            lora_dropout=0.1,
            bias="none",
        )
    lora_encoder = Lora_Encoder(num_layers=6, lora_config=lora_config)
    print(lora_encoder)
