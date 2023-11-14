import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
import torchvision.transforms as transforms

def load_model():
    # load vgg19 model
    model = models.vgg19(pretrained=True)

    return model

class CustomVGG(nn.Module):
    def __init__(self):
        super(CustomVGG, self).__init__()
        # load original vgg19 model
        vgg19 = load_model()
        pretrained_features = vgg19.features

        # name model as features because first part of pretrained vgg19 is named features
        self.features = nn.Sequential()
        prev_layer = 0
        for module in pretrained_features:
            if isinstance(module, nn.MaxPool2d):
                # replace MaxPool2d in vgg19 using AvgPool2d
                module = nn.AvgPool2d(kernel_size=module.kernel_size, stride=module.stride, padding=module.padding)

            self.features.add_module(str(len(self.features)), module)

            # load weight in pretrained vgg19
            if not (isinstance(module, nn.MaxPool2d) or isinstance(module, nn.ReLU)):
                self.features[-1].load_state_dict(pretrained_features[int(prev_layer)].state_dict())
                
            prev_layer += 1

    def forward(self, x):
        x = self.features(x)
        return x

def Get_preprocess(crop_size):
    crop_size = (crop_size[0], crop_size[1])
    preprocess = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess