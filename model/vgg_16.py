import torch
import torch.nn as nn
import torch.nn.functional as F

from model.hybrid_CNN import Hybrid_Conv2d


model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}

model_path = "model/vgg16_pretrained.pth"


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True): # change to binary classifier
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, Hybrid_Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# M means MaxPool
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    return model


def vgg16(pretrained=False, progress=True, **kwargs):
    r""" (CUSTOMIZED) VGG 16-layer model (configuration "D")
    Takes in the cov paramter and forward function is customized
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)



class MyVGG16(nn.Module):
    def __init__(self):
        super(MyVGG16, self).__init__()
        # load pytorch vgg16 with pretrained weights
        vgg = vgg16(pretrained=True)

        # set the three blocks you need for forward pass
        # remove the first conv layer + relu from the feature extractor
        self.features = vgg.features[2:]
        self.avgpool = vgg.avgpool
        self.classifier = vgg.classifier
        
        # my precious baby hybrid layers
        self.hybrid_conv1 = Hybrid_Conv2d(3, 16, kernel_size=(64, 3, 3, 3), cov=0) 
        self.hybrid_conv2 = Hybrid_Conv2d(3, 16, kernel_size=(64, 3, 3, 3), cov=1)
        
    # Set your own forward pass
    def forward(self, x, cov):
        if cov==0:
            x = F.relu(self.hybrid_conv1(x))
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
        elif cov==1:
            x = F.relu(self.hybrid_conv2(x))
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x




# # https://discuss.pytorch.org/t/how-can-i-replace-the-forward-method-of-a-predefined-torchvision-model-with-my-customized-forward-function/54224/6
# class MyVGG16(VGG):
#     def __init__(self, features):
#         super(MyVGG16, self).__init__(features)
#         self.features = features [2:]
#         # my precious hybrid layers
#         self.hybrid_conv1 = Hybrid_Conv2d(3, 16, kernel_size=(16, 3, 3, 3), cov=0) 
#         self.hybrid_conv2 = Hybrid_Conv2d(3, 16, kernel_size=(16, 3, 3, 3), cov=1)

#     def forward(self, x, cov):
#         if cov==0:
#             x = F.relu(self.hybrid_conv1(x))
#             x = self.features(x)
#             x = self.avgpool(x)
#             x = torch.flatten(x, 1)
#             x = self.classifier(x)
#             return x
#         elif cov==1:
#             x = F.relu(self.hybrid_conv2(x))
#             x = self.features(x)
#             x = self.avgpool(x)
#             x = torch.flatten(x, 1)
#             x = self.classifier(x)
#             return x
