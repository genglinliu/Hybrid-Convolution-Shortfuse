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

## Experiments Without Hybrid Layers
1. First tried a resnet-18 pretrained on ImageNet and froze the layers except for the last one. Had about 63% validation accuracy. 
2. Then replaced pretrained resnet with a pretrained VGG-16. Trained on one epoch over the training set and the val accuracy was about 73%. The transfer learning does not perform well for either model. 
3. Tried to train vgg-16 from scratch (without pretraining on ImageNet). High memory use and could not perform this locally. Had to use the cluster to compute but the accuracy was horrible - barely over 50%
4. We finally took a pre-trained VGG-16 but re-trained it on CelebA - aka no parameter was frozen at the beginning and every layer was retrained. This gave the best performance - validation accuracy about 78-79% on attrativeness attribute prediction. `batchsize=8`

## Hybrid Conv2d Layer
One of the core feature of this project is the implementation of a customized convolutional layer. 
Unlike the vanilla convolutional layers, our hybrid conv layer takes a structured covariate as parameter, and this cov variable will activate the learning of certain weights when its value is non-zero.

Pytorch layers are implemented as classes that extend `nn.Module`, and so we defined two weight matrices and introduce the covariate as a single scaler whose value depends on the sample that goes through. The only other thing we had to do is defining the forward pass. 

ex. if the current training image has a label "male", then the conv parameter of the hybrid layer will be 1 *for that sample*

This will be adapted to 3D tasks (brain MRIs) if needed. If 2D works, 3D should work as well.

## Experiments With Hybrid Layers
First experiment terrible results - 0.52 val acc [with normal weight initialization]

 ## 3/14 update
 The two-layer hybrid cnn now works! It took some tricks to make this network work:
  - Had to define different forward passes with different covariate values
  - At each iteration only one image can pass through the network so batch size can only be 1
    - If we don't do this we'd have a `RuntimeError: boolean value of Tensor with more than one value is ambiguous`
    - because we'd have multiple images but only one integer for the cov value
  - No `nn.Sequential` containers because we need to customize forward passes; but that's fine because we're not really doing a two layer net anyway

The TODOs remain the same for now; a few additional things:

## TODO
 - Modify the vgg net and replace the early layers with the hybrid conv layers [done]
 - try plot the loss *while* training
 - saving checkpoint code should be changed - just name the checkpoint with the `experiment_name`
 - log training loss into a log file [done]
 - save/load the dataloader to save time `torch.save(dataloader_obj, 'dataloader.pth')`
 - Have this extendable to 3D conv layers because we're eventually going to work with ADNI images
 - add F1 score as a performance measure

## 3/21 update
 - So after researching online I found that it's not the best idea to pull the pytorch source code and modify it directly
   rather, we should create our own model that extends VGG or just nn.Module and 

## 3/23 update
 - there are most likely something wrong with the other parts of the code after running three experiements on titanX
 - set s=0 for all cov for vgg with hybrid layers, batchsize=1 - 0.52
 - regular vgg16 with batchsize=1  => val acc = 0.48
 - regualr vgg16 with batchsize=8  => val acc = 0.52 (exactly the same as exp1 which is even more strange)
   - this used to be 0.78-0.79
 - loss oscillates between 0.6x-0.7x
