from transformers import ViTImageProcessor, ViTModel, ViTConfig
import copy
from torch import nn
from peft import LoraConfig, get_peft_model


class My_Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def deleteEncodingLayers(model, num_layers_to_keep):  # must pass in the full bert model
    oldModuleList = model.base_model.encoder.layer
    newModuleList = nn.ModuleList()

    # Now iterate over all layers, only keepign only the relevant layers.
    for i in range(num_layers_to_keep):
        newModuleList.append(oldModuleList[i])

    # create a copy of the model, modify it with the new list, and return
    copyOfModel = copy.deepcopy(model)
    copyOfModel.base_model.encoder.layer = newModuleList
    copyOfModel.base_model.layernorm = My_Identity()
    copyOfModel.base_model.pooler = My_Identity()

    return copyOfModel


class Lora_Encoder:
    def __init__(self, num_layers=6):
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.vit_config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k", output_hidden_states=True)
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k', config=self.vit_config)
        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
        )
        lora_model = get_peft_model(self.model, peft_config=self.lora_config)
        self.model = deleteEncodingLayers(lora_model, num_layers)

    def forward(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 1:, :], self.model.embeddings.position_embeddings[:, 1:, :]
