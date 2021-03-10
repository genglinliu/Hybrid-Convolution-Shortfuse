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
    
    
    
class ConvNet(nn.Module):
    """
    Simple two-layer CNN 
    """
    def __init__(self):
        super(ConvNet, self).__init__()
        # TODO: define the layers of the network
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3), ############## change this to hybrid and add cov param ################
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            nn.Flatten()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(387200, 128),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(128, 2) # binary classification
        )

    def forward(self, x):
        out = self.layer1(x)
        print(out.shape)
        out = self.layer2(out)
        return out
    
    
    
class TwoLayerCNN(nn.Module):
    """ 
    model from pytorch tutorial site 
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """
    def __init__(self):
        super(TwoLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(359552, 120) # change made here
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 359552)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x