import torch
from torch import nn
import torchvision.models as models
from torchvision.models import VGG19_Weights

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ["0", "5", "10", "19", "28"]
        self.model = models.vgg19(weights=VGG19_Weights.DEFAULT).features[:29]
        # freeze the vgg19
        for p in self.model.parameters():
            if p.requires_grad:
                p.requires_grad_(False)
    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features
    
def content_loss(vgg, contents, gens):
    """
    #### input ####
    vgg     : the pretrained and feature selected vgg19 model, FROZEN
    contents: the original content images, in shape [B, 3, 224, 224]
    gens    : the generated images, in shape [B, 3, 224, 224]
    
    #### output ####
    loss_c  : a scalar indicating the content loss of the input batch
    
    """
    features_c = vgg(contents)
    features_g = vgg(gens)
    # features is a 5-element list, each element is in shape [B, C, H, W], with different C, H, W for each element
    loss_c = 0.
    for feature_c, feature_g in zip(features_c, features_g):
        loss_c += torch.mean((feature_c - feature_g) ** 2)
    return loss_c 
    
def calc_mean_std(feat):
    """
    #### input ####
    feat    : a feature output of a certain vgg layer, in shape [B, C, H, W]
    #### output ####
    mean    : in shape [B, C]
    std     : in shape [B, C]
    
    """
    B, C, H, W = feat.shape
    mean = feat.view(B, C, -1).mean(dim=2)
    std = feat.view(B, C, -1).std(dim=2)
    return mean, std

def style_loss(vgg, styles, gens):
    """perceptual loss
    #### input ####
    vgg     : the pretrained and feature selected vgg19 model, FROZEN
    styles  : the original style images, in shape [B, 3, 224, 224]
    gens    : the generated images, in shape [B, 3, 224, 224]
    
    #### output ####
    loss_s  : a scalar indicating the style loss of the input batch
    
    """
    features_s = vgg(styles)
    features_g = vgg(gens)
    # features is a 5-element list, each element is in shape [B, C, H, W], with different C, H, W for each element
    loss_s = 0.
    for feature_s, feature_g in zip(features_s, features_g):
        mean_s, std_s = calc_mean_std(feature_s)
        mean_g, std_g = calc_mean_std(feature_g)
        loss_s += torch.mean((mean_s - mean_g) ** 2) + torch.mean((std_s - std_g) ** 2)
    return loss_s    
    
# def style_loss(vgg, styles, gens):
#     """Gram matrix method
#     #### input ####
#     vgg     : the pretrained and feature selected vgg19 model, FROZEN
#     styles  : the original style images, in shape [B, 3, 224, 224]
#     gens    : the generated images, in shape [B, 3, 224, 224]
    
#     #### output ####
#     loss_s  : a scalar indicating the style loss of the input batch
    
#     """    
#     features_s = vgg(styles)
#     features_g = vgg(gens)
#     # features is a 5-element list, each element is in shape [B, C, H, W], with different C, H, W for each element
#     loss_s = 0.
#     for feature_s, feature_g in zip(features_s, features_g):
#         B, C, H, W = feature_s.shape
#         feat_s_flat = feature_s.view(B, C, -1) # [B, C, H*W]
#         S = feat_s_flat @ feat_s_flat.transpose(1,2) # [B, C, C]
#         feat_g_flat = feature_g.view(B, C, -1) # [B, C, H*W]
#         G = feat_g_flat @ feat_g_flat.transpose(1,2) # [B, C, C]    
#         loss_s += (1 / (4 * B * H**2 * W**2 * C**2)) * torch.sum((G - S) ** 2)
#     return loss_s
    

def identity_loss_1(Ic, Icc, Is, Iss):
    """
    #### input ####
    Ic          : the original content images, in shape [B, 3, 224, 224]
    Icc         : the generated images using Ic as both content and style, in shape [B, 3, 224, 224]
    Is          : the original style images, in shape [B, 3, 224, 224]
    Iss         : the generated images using Is as both content and style, in shape [B, 3, 224, 224]

    #### output ####
    id_loss_1   : a scalar indicating the reconstruction loss 
    
    """
    id_loss_1 = torch.mean((Ic - Icc) ** 2) + torch.mean((Is - Iss) ** 2)
    return id_loss_1

def identity_loss_2(vgg, Ic, Icc, Is, Iss):
    """
    #### input ####
    vgg         : the pretrained and feature selected vgg19 model, FROZEN
    Ic          : the original content images, in shape [B, 3, 224, 224]
    Icc         : the generated images using Ic as both content and style, in shape [B, 3, 224, 224]
    Is          : the original style images, in shape [B, 3, 224, 224]
    Iss         : the generated images using Is as both content and style, in shape [B, 3, 224, 224]

    #### output ####
    id_loss_2   : a scalar indicating the mean feature loss of the pairs of (Ic, Icc) and (Is, Iss)
    
    """    
    features_c = vgg(Ic)
    features_cc = vgg(Icc)
    features_s = vgg(Is)
    features_ss = vgg(Iss)
    # features is a 5-element list, each element is in shape [B, C, H, W], with different C, H, W for each element
    id_loss_2 = 0.
    for feature_c, feature_cc, feature_s, feature_ss in zip(features_c, features_cc, features_s, features_ss):
        id_loss_2 += torch.mean((feature_c - feature_cc) ** 2) + torch.mean((feature_s - feature_ss) ** 2)
    return id_loss_2     
    
def calc_total_loss(vgg, Ig, Ic, Icc, Is, Iss, lambda_c, lambda_s, lambda_id1, lambda_id2):
    """
    #### input ####
    vgg         : the pretrained and feature selected vgg19 model, FROZEN
    Ig          : the generated images using Ic and Is, in shape [B, 3, 224, 224]
    Ic          : the original content images, in shape [B, 3, 224, 224]
    Icc         : the generated images using Ic as both content and style, in shape [B, 3, 224, 224]
    Is          : the original style images, in shape [B, 3, 224, 224]
    Iss         : the generated images using Is as both content and style, in shape [B, 3, 224, 224]
    lambda_c    : the coefficient of content loss
    lambda_s    : the coefficient of style loss
    lambda_id1  : the coefficient of id loss 1
    lambda_id2  : the coefficient of id loss 2
    #### output ####
    loss        : a scalar indicating the total loss
    loss_c      : the content loss
    loss_s      : the style loss
    loss_id1    : the identity loss 1
    loss_id2    : the identity loss 2
    
    """      
    loss_c = content_loss(vgg, Ic, Ig)
    loss_s = style_loss(vgg, Is, Ig)
    loss_id1 = identity_loss_1(Ic, Icc, Is, Iss)
    loss_id2 = identity_loss_2(vgg, Ic, Icc, Is, Iss)
    loss = lambda_c*loss_c + lambda_s*loss_s + lambda_id1*loss_id1 + lambda_id2*loss_id2
    return loss, loss_c, loss_s, loss_id1, loss_id2
    