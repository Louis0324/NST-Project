import torch

def content_loss(gen_outputs, content_outputs, name_2_index, chosen_layer="conv4_2"):
    
    def Compute_Loss(ori_img, gen_img):
        """
        Args:
            ori_img (_type_): original image
            gen_img (_type_): generated image
        """
        
        #print("ori_img shape: ",ori_img.shape)
        #print("gen_img shape: ",gen_img.shape)
        
        channel, height, width = ori_img.shape
        
        # N is number of filters
        N = channel
        # M is size of channel
        M = height * width
        
        P = ori_img
        X = gen_img
        # P and X are both in shape: (channel, height, width)
        
        result = (1 / (4 * N * M)) * torch.sum((P - X)**2)
        return result
    
    key = chosen_layer
    
    ori_img = content_outputs[key]
    gen_img = gen_outputs[key]
        
    # here output of certain layer is in shape
    # [batch_size, channels, height, width]
    # and in fact batch_size = 1 in this task
    # so we should squeeze the first dimension
    ori_img = ori_img.squeeze(0)
    gen_img = gen_img.squeeze(0)
    
    loss = Compute_Loss(ori_img, gen_img)
    return loss


def style_loss(gen_outputs, style_outputs, name_2_index, weight_dict):

    def Make_Gram_Matrix(R, N, M):
        """
        The gram matrix G.
        
        R(response): (channel, height, width)
        F: (channel, height*width)
        G = F @ F.t
        G: (channel, channel)
        
        """
        
        F = R.view(N, M)
        G = torch.matmul(F, F.t())
        
        return G
    
    def Compute_Loss(ori_img, gen_img):
        """
        Args:
            ori_img (_type_): original image
            gen_img (_type_): generated image
        """
        
        channel, height, width = ori_img.shape
        
        # N is number of filters
        N = channel
        # M is size of channel
        M = height * width
        
        # A is Gram matrix(style representation) of original image
        A = Make_Gram_Matrix(ori_img, N, M)
        # G is Gram matrix(style representation) of generated image
        G = Make_Gram_Matrix(gen_img, N, M)
        
        result = (1 / (4 * N**2 * M**2)) * torch.sum((G - A)**2)
        return result
    
    loss = 0 
    for key in name_2_index.keys():
        ori_img = style_outputs[key]
        gen_img = gen_outputs[key]
        
        # here output of certain layer is in shape
        # [batch_size, channels, height, width]
        # and in fact batch_size = 1 in this task
        # so we should squeeze the first dimension
        ori_img = ori_img.squeeze(0)
        gen_img = gen_img.squeeze(0)
        
        layer_loss = Compute_Loss(ori_img, gen_img)
        weight = weight_dict[key]
        loss += weight * layer_loss
        
    return loss
        
        
def Total_loss(content_loss, style_loss, alpha=1, beta=1000):
    return alpha * content_loss + beta * style_loss

def Get_loss(gen_outputs, style_outputs, content_outputs,
             style_name_2_index, style_weight_dict, cont_name_2_index,
             alpha = 1, beta = 1000):
    
    cont_loss = content_loss(gen_outputs, content_outputs, cont_name_2_index, "conv4_2")
    sty_loss = style_loss(gen_outputs, style_outputs, style_name_2_index, style_weight_dict)
    
    total_loss = Total_loss(cont_loss, sty_loss, alpha, beta)
    return total_loss

class CustomLoss(torch.nn.Module):
    def __init__(self, style_name_2_index, style_weight_dict, cont_name_2_index,
             alpha = 1, beta = 1000):
        super(CustomLoss, self).__init__()
        self.style_name_2_index = style_name_2_index
        self.style_weight_dict = style_weight_dict
        self.cont_name_2_index = cont_name_2_index
        self.alpha = alpha
        self.beta = beta

    def forward(self, gen_outputs, style_outputs, content_outputs):
        loss = Get_loss(gen_outputs, style_outputs, content_outputs,
                         self.style_name_2_index, self.style_weight_dict,
                         self.cont_name_2_index, self.alpha, self.beta)
        return loss