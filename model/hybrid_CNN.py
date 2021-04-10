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
# The hybrid Conv2d layer
###################

class Hybrid_Conv2d(nn.Module):
    """    
    (self, channel_in, channel_out, kernel_size, cov, stride=1, padding=0)
    kernel_size are 4d weights: (out_channel, in_channel, height, width)
    """    
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0):
        super(Hybrid_Conv2d, self).__init__()
        self.kernel_size = kernel_size # 4D weight (out_channel, in_channel, height, width)
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.stride = stride
        self.padding = padding
        # self.cov = cov  # cov vector of shape = (minibatch,)

        self.W_0 = nn.Parameter(torch.randn(kernel_size), requires_grad=True)
        self.W_1 = nn.Parameter(torch.randn(kernel_size), requires_grad=True)        
        self._initialize_weights()
        
    # weight initialization
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.W_0, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.W_1, mode='fan_out', nonlinearity='relu')
 
    def forward(self, x, cov):
        # input x is of shape = (minibatch, channel=3, width, height) e.g. (32, 3, 224, 224)
        outputs = []
        for i in range(cov.shape[0]): # s_l is the scalar covariate per data point
            kernel = self.W_0 + torch.mul(self.W_1, cov[i]) # debug: could replace cov[i] with 0
            x_i = torch.unsqueeze(x[i], 0) # (3, 224, 224) -> (1, 3, 224, 224) for 4d weight shape matching
            out = F.conv2d(x_i, kernel, stride=self.stride, padding=self.padding)
            outputs.append(out) 
            
        outputs = torch.cat(outputs)
        return outputs