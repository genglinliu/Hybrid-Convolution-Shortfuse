import torch
import torch.nn as nn
import torch.nn.functional as F


class Hybrid_conv2d(nn.Module):
    def __init__(self):
        super(Hybrid_conv2d, self).__init__()
        kernel = [[0.03797616, 0.044863533, 0.03797616],
                  [0.044863533, 0.053, 0.044863533],
                  [0.03797616, 0.044863533, 0.03797616]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
 
    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=2)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=2)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight, padding=2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x
    
    
    
    
class Conv2d_symmetric(nn.Module):
    def __init__(self):
    super(Conv2d_simple, self).__init__()

    self.a = nn.Parameter(torch.randn(1))
    self.b = nn.Parameter(torch.randn(1))
    self.c = nn.Parameter(torch.randn(1))


    self.bias = None
    self.stride = 2
    self.padding = 1
    self.dilation = 1
    self.groups = 1



    def forward(self, input):

        #in case we use gpu we need to create the weight matrix there
        device = self.a.device

        weight = torch.zeros((1,1,3,3)).to(device)
        weight[0,0,0,0] += self.c[0]
        weight[0,0,0,1] += self.b[0]
        weight[0,0,0,2] += self.c[0]
        weight[0,0,1,0] += self.b[0]
        weight[0,0,1,1] += self.a[0]
        weight[0,0,1,2] += self.b[0]
        weight[0,0,2,0] += self.c[0]
        weight[0,0,2,1] += self.b[0]
        weight[0,0,2,2] += self.c[0]


        # print("weight= ", weight)
        # print("inout = ", input)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        
        
########################################################################################
  
  
  
      
import torch.nn.functional as F
import torch

def gabor_fn(kernel_size, channel_in, channel_out, sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma    # [channel_out]
    sigma_y = sigma.float() / gamma     # element-wize division, [channel_out]

    # Bounding box
    nstds = 3 # Number of standard deviation sigma
    xmax = kernel_size // 2
    ymax = kernel_size // 2
    xmin = -xmax
    ymin = -ymax
    ksize = xmax - xmin + 1
    y_0 = torch.arange(ymin, ymax+1)
    y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1).float()
    x_0 = torch.arange(xmin, xmax+1)
    x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize).float()   # [channel_out, channelin, kernel, kernel]

    # Rotation
    # don't need to expand, use broadcasting, [64, 1, 1, 1] + [64, 3, 7, 7]
    x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
    y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))

    # [channel_out, channel_in, kernel, kernel]
    gb = torch.exp(-.5 * (x_theta ** 2 / sigma_x.view(-1, 1, 1, 1) ** 2 + y_theta ** 2 / sigma_y.view(-1, 1, 1, 1) ** 2)) \
         * torch.cos(2 * math.pi / Lambda.view(-1, 1, 1, 1) * x_theta + psi.view(-1, 1, 1, 1))

    return gb


class GaborConv2d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0):
        super(GaborConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.stride = stride
        self.padding = padding

        self.Lambda = nn.Parameter(torch.rand(channel_out), requires_grad=True)
        self.theta = nn.Parameter(torch.randn(channel_out) * 1.0, requires_grad=True)
        self.psi = nn.Parameter(torch.randn(channel_out) * 0.02, requires_grad=True)
        self.sigma = nn.Parameter(torch.randn(channel_out) * 1.0, requires_grad=True)
        self.gamma = nn.Parameter(torch.randn(channel_out) * 0.0, requires_grad=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        theta = self.sigmoid(self.theta) * math.pi * 2.0
        gamma = 1.0 + (self.gamma * 0.5)
        sigma = 0.1 + (self.sigmoid(self.sigma) * 0.4)
        Lambda = 0.001 + (self.sigmoid(self.Lambda) * 0.999)
        psi = self.psi

        kernel = gabor_fn(self.kernel_size, self.channel_in, self.channel_out, sigma, theta, Lambda, psi, gamma)
        kernel = kernel.float()   # [channel_out, channel_in, kernel, kernel]

        out = F.conv2d(x, kernel, stride=self.stride, padding=self.padding)

        return out