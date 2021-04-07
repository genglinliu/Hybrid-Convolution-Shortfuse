import torch
import torch.nn as nn
import torch.nn.functional as F


"""
A d-dim covaraite vector S as input

K_l = W_0 + W_1 * S_l 
where S_l is one of the scalar entries of S. it is either 0 (female) or 1 (male)

When S is a batch input, kernel param k_l still needs to be updated per data point (image)

Solution: For each minibatch of size N, with the kernel param W0 and W1, 
first convolve each data point in the minibatch with either W_0 or W_0+W_1 (depend on the covariate), 
then you concat all the output of N convolution, do batchnorm

"""

###################
# Our hybrid layer
###################

class Hybrid_Conv2d(nn.Module):
    """    
    (self, channel_in, channel_out, kernel_size, cov, stride=1, padding=0)
    kernel_size are 4d weights: (out_channel, in_channel, height, width)
    """    
    def __init__(self, channel_in, channel_out, kernel_size, cov, stride=1, padding=0):
        super(Hybrid_Conv2d, self).__init__()
        self.kernel_size = kernel_size # 4D weight (out_channel, in_channel, height, width)
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.stride = stride
        self.padding = padding
        self.cov = cov  # cov vector of shape = (minibatch,)

        self.W_0 = nn.Parameter(torch.randn(kernel_size), requires_grad=True)
        self.W_1 = nn.Parameter(torch.randn(kernel_size), requires_grad=True)        
        self._initialize_weights()
        
    # weight initialization
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.W_0, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.W_1, mode='fan_out', nonlinearity='relu')
 
    def forward(self, x, cov):
        # input x is of shape = (minibatch, channel=3, width, height) e.g. (32, 3, 224, 224)
        cov = self.cov # (minibatch,)
        W_0 = self.W_0
        W_1 = self.W_1
        
        outputs = []
        for s_l in cov: # s_l is the scalar covariate per data point
            kernel = W_0 + torch.mul(W_1, s_l)       
            out = F.conv2d(x, kernel, stride=self.stride, padding=self.padding)
            outputs.append(out) 
        
        return outputs
    
    
    
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