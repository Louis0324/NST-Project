from transformers import ViTImageProcessor, ViTModel, ViTConfig
import copy
from torch import nn
from peft import LoraConfig, get_peft_model


def deleteEncodingLayers(model, num_layers):
    oldModuleList = model.base_model.encoder.layer
    newModuleList = nn.ModuleList()

    for i in range(num_layers):
        newModuleList.append(oldModuleList[i])

    copyOfModel = copy.deepcopy(model)
    copyOfModel.base_model.encoder.layer = newModuleList

    return copyOfModel


vit_config = ViTConfig(hidden_size=768,
                       num_hidden_layers=12,
                       num_attention_heads=12,
                       intermediate_size=3072,
                       hidden_act='gelu',
                       hidden_dropout_prob=0.0,
                       attention_probs_dropout_prob=0.0,
                       initializer_range=0.02,
                       layer_norm_eps=1e-12,
                       image_size=224,
                       patch_size=16,
                       num_channels=3,
                       qkv_bias=True,
                       encoder_stride=16,
                       )


class Lora_Encoder:
    def __init__(self, num_layers):
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
        )
        lora_model = get_peft_model(self.model, self.config)
        self.model = deleteEncodingLayers(lora_model, num_layers)

    def forward(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.last_hidden_state
