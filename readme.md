CelebA facial attribute prediction - attractiveness

updated with python script because it's harder to set up jupyter kernel using the gpu cluster.

## The Problem
Predict a binary facial attribute in the CelebA dataset using deep learning. Investigate whether shortfuse will improve performance.

## Model
Pretrained VGG-16 with the last linear layer modified for binary classification.
```python
model = models.vgg16_bn(pretrained=True)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, 2)
```

## Experiments
1. First tried a resnet-18 pretrained on ImageNet and froze the layers except for the last one. Had about 63% validation accuracy. 
2. Then replaced pretrained resnet with a pretrained VGG-16. Trained on one epoch over the training set and the val accuracy was about 73%. The transfer learning does not perform well for either model. 
3. Tried to train vgg-16 from scratch (without pretraining on ImageNet). High memory use and could not perform this locally. Had to use the cluster to compute but the accuracy was horrible - barely over 50%
4. We finally took a pre-trained VGG-16 but re-trained it on CelebA - aka no parameter was frozen at the beginning and every layer was retrained. This gave the best performance - validation accuracy about 78-79% on attrativeness attribute prediction. 

## TODO
 - Modify the initial layers of vgg and proceed with shortfuse