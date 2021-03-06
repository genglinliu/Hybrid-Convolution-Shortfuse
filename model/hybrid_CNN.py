import torch
import torch.nn as nn
import torch.nn.functional as F


"""
You got a d-dim covaraite vector as input
K_l = W_0 + W_1 * S_l

S_0 = male (1)
S_1 = female (0)

for i, (images, labels) in enumerate(train_loader):
    labels = labels[:, 2] # attractiveness label

The rest is just tensor computation

on every iteration, only one image sample passes through the hybrid layer
so we pass in this one scaler (1 or 0) to the hybrid layer as a parameter
We define s_l as a scaler, so the '1' samples will make w_1 activate

K_l = W_0 + W_1 * S_l

"""

class Hybrid_Conv2d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0, cov=0):
        super(Hybrid_Conv2d, self).__init__()
        self.kernel_size = kernel_size # tuple, ex. (3, 3)
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.stride = stride
        self.padding = padding
        self.cov = cov
        
        self.W_0 = nn.Parameter(torch.randn(kernel_size), requires_grad=True)
        self.W_1 = nn.Parameter(torch.randn(kernel_size), requires_grad=True)
 
    def forward(self, x):
        
        cov = self.cov 
        W_0 = self.W_0
        W_1 = self.W_1
        
        kernel = W_0 + torch.mul(W_1, cov)
        out = F.conv2d(x, kernel, stride=self.stride, padding=self.padding)
        return out