import torch
import torch.nn as nn
import torch.nn.functional as F


"""
[deprecated - only process covariate as scalar values and processes images one by one]

You got a d-dim covaraite vector as input
K_l = W_0 + W_1 * S_l

S_0 = male (1)
S_1 = female (0)

for i, (images, labels) in enumerate(train_loader):
    labels = labels[:, 2] # attractiveness label

The rest is just tensor computation

on every iteration, only one image sample passes through the hybrid layer
so we pass in this one scaler (1 or -1) to the hybrid layer as a parameter
We define s_l as a scaler, so the '1' samples will make w_1 activate

K_l = W_0 + W_1 * S_l

When S_l is a batch input, kernel param k_l still needs to be updated per data point (image)
So we need to handle that
"""

###################
# Our hybrid layer
###################

class Hybrid_Conv2d(nn.Module):
    """    
    (self, channel_in, channel_out, kernel_size, stride=1, padding=0, cov=0)
    kernel_size are 4d weights: (out_channel, in_channel, height, width)
    """    
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0, cov=0):
        super(Hybrid_Conv2d, self).__init__()
        self.kernel_size = kernel_size # 4D weight (out_channel, in_channel, height, width)
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.stride = stride
        self.padding = padding
        self.cov = cov  # currently a scalar; cov vector of shape = (minibatch,)
        
        # initialization: gaussian random
        self.W_0 = nn.Parameter(torch.randn(kernel_size), requires_grad=True)
        self.W_1 = nn.Parameter(torch.randn(kernel_size), requires_grad=True)
        
        # N = cov.shape[0] # length of covariate vector is the batchsize
        # weights = []
        # for _ in range(N):
        #     weight = nn.Parameter(torch.randn(15, 3, 5, 5))
        #     weights.append(weight)
        
        self._initialize_weights()
        
    # weight initialization
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.W_0, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.W_1, mode='fan_out', nonlinearity='relu')
 
    def forward(self, x):
        # input x is of shape = (minibatch, channel=3, width, height) e.g. (32, 3, 224, 224)
        cov = self.cov # (minibatch,)
        W_0 = self.W_0
        W_1 = self.W_1
        
        kernel = W_0 + torch.mul(W_1, cov)
        out = F.conv2d(x, kernel, stride=self.stride, padding=self.padding)
        return out
    
    
    
# experiment with two layer CNN
    
class HybridConvNet(nn.Module):
    """
    Simple two-layer CNN with hybrid layer
    """
    def __init__(self): 
        super(HybridConvNet, self).__init__()    
        self.hybrid_conv1 = Hybrid_Conv2d(3, 16, kernel_size=(16, 3, 3, 3), cov=0) 
        self.hybrid_conv2 = Hybrid_Conv2d(3, 16, kernel_size=(16, 3, 3, 3), cov=1)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(387200, 128)
        self.fc2 = nn.Linear(128, 2) # binary classification
        
    def forward(self, x, cov):
        if cov==0:
            x = F.relu(self.hybrid_conv1(x))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(1, -1) # flatten
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        elif cov==1:
            x = F.relu(self.hybrid_conv2(x))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(1, -1) # flatten
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        return x
    
    
    
class ConvNet_v1(nn.Module):
    """
    Simple two-layer CNN with sequential container
    """
    def __init__(self):
        super(ConvNet_v1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(387200, 128),
            nn.ReLU(),
            nn.Linear(128, 2) 
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out
    
class ConvNet_v2(nn.Module):
    """
    Simple two-layer CNN with no Sequential container
    """
    def __init__(self): 
        super(ConvNet_v2, self).__init__()    
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(387200, 128)
        self.fc2 = nn.Linear(128, 2) 
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(1, -1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x