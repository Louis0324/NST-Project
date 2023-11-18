import torch
from torch import nn

vgg_model = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

def load_vgg(path = '../files/vgg_pretrained.pth'):
    """
    load pretrained vgg for computing loss
    """
    vgg = vgg_model
    vgg.load_state_dict(torch.load(path))
    vgg = nn.Sequential(*list(vgg.children())[:44])
    return vgg

class Loss_Computer():
    """
    class for computing loss
    """
    def __init__(self):
        network = load_vgg()
        feature_layers = list(network.children())
        self.ft_1 = nn.Sequential(*feature_layers[:4])  # input -> relu1_1
        self.ft_2 = nn.Sequential(*feature_layers[4:11])  # relu1_1 -> relu2_1
        self.ft_3 = nn.Sequential(*feature_layers[11:18])  # relu2_1 -> relu3_1
        self.ft_4 = nn.Sequential(*feature_layers[18:31])  # relu3_1 -> relu4_1
        self.ft_5 = nn.Sequential(*feature_layers[31:44])  # relu4_1 -> relu5_1
        
        for name in ['ft_1', 'ft_2', 'ft_3', 'ft_4', 'ft_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.mse = nn.MSELoss()

    def extract_features(self, input):
        features = [input]
        for i in range(5):
            func = getattr(self, 'ft_{:d}'.format(i + 1))
            features.append(func(features[-1]))
        return features[1:]

    def content_loss(self):
        return
    
    def style_loss(self):
        return

if __name__ == '__main__':
    input = torch.randn((8, 3, 224, 224))
    loss_computer = Loss_Computer()
    fts = loss_computer.extract_features(input)
    

