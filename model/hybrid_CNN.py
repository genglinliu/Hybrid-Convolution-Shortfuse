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
    """    
    (self, channel_in, channel_out, kernel_size, stride=1, padding=0, cov=0)
    """    
    def __init__(self, channel_in, channel_out, kernel_size, cov, stride=1, padding=0, ):
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
    
    
    
class ConvNet(nn.Module):
    """
    Simple two-layer CNN 
    https://discuss.pytorch.org/t/forward-pass-with-different-weights/9068
    """
    def __init__(self): # changed here
        super(ConvNet, self).__init__()
        # TODO: define the layers of the network
        self.layer1 = nn.Sequential(
            Hybrid_Conv2d(3, 16, kernel_size=(3, 3), cov=cov), # changed here
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(387200, 128), # out.shape after layer1 in forward
            nn.ReLU(),
            nn.Linear(128, 2) # binary classification
        )

    def forward(self, x, cov):
        if cov==0:
            out = self.layer1(x, cov)
            out = self.layer2(out)
        elif cov==1:
            out = self.layer1(x, cov)
            out = self.layer2(out)
        return out