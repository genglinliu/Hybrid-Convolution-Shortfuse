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
 - set s=0 for all cov for vgg with hybrid layers, batchsize=1 => val acc = 0.52
 - regular vgg16 with batchsize=1  => val acc = 0.48
 - regualr vgg16 with batchsize=8  => val acc = 0.52 (exactly the same as exp1 which is even more strange)
   - this used to be 0.78-0.79
 - loss oscillates between 0.6x-0.7x

## 3/24 update
I thought the script had problems so I went back to commit `bfa4b43` and checked out in another branch and pulled the old `celeb_a.py` script.
Then I ran the following experiments:

exp1: vgg16_bn  batchsize=16  old script  [val acc = 0.783] - finished under 1 hour
exp2: vgg16     batchsize=16  old script  [val acc = 0.5200583882820758] - ocsillating training loss
exp3: vgg16     bacthsize=1   old script  [val acc = 0.5200583882820758] - unstable training
exp4: vgg16_bn  batchsize=1   old script  [val acc = 0.4799416117179242] -  gradient vanished
exp5: vgg16_bn  batchsize=16  new script  [val acc = 0.791]

And so I realized the problem was probably that I gave up on vgg16_bn when I thought my hybrid layer couldn't handle minibacthes.
From the experiments we can see how important BatchNorm is for our task. If we have pure SGD then we can't use BN.
Now I know that it is crucial to first enable the hybrid layers 

So right now the **TODO** is make hybrid layers take batch inputs and run it with vgg16_bn

ps to debug locally maybe you can load just the val set AS training set? To figure out shapes and stuff
That way we may be able to avoid the CUDA out of memory error

0.4799416117179242 and 0.5200583882820758 - that's just the model predicting all 0's or 1's. Why？
We now see how much BatchNorm helps stablize training - we will try more batchsizes and find the best one

### TODOs
 - make hybrid layers take batch inputs and run it with vgg16_bn
   - You're going to find this in F.Conv2d (input, weight, bias, ...) where input = (minibatch,in_channels,iH,iW)
 - add f1 score as metrics
 - at one point you will have to revisit the eval code, maybe add cross validation etc

More exp:
 exp1: batchsize=32 => val acc = 0.806
 exp2: batchsize=64 => val acc = 0.805

## 4/6 updates

TODOs: For each minibatch of size N, with the kernel param W0 and W1, you first convolve each data point in the minibatch with either W0 or W0+W1 (depend on the covariate), then you concat all the output of N convolution, do batchnorm

## 4/9 update

So we made it that the hybrid layers could take batch inputs; here is the stack of logic:
1. In forward pass of `CelebA.py`, we have `outputs = model(images, cov_attr)`, where model is an instance of `MyVGG()` and a minibatch of images and their corresponding covariates (together as a 1d tensor) are passed in
2. In `vgg16.py`, class MyVGG() first calls `vgg16_bn(pretrained=True)` to instantiate a pretrained vgg16_bn, then replace the first conv layer with a hybrid conv layer. In the forward pass, image and cov enter the `hybrid_conv2d` layer together
3. In `model/hybrid_CNN.py`, we have class `Hybrid_Conv2d` and here's how this layer works:
  - Conv kernels gets updated per minibatch
  - Each conv kernel consists of two weight tensors, `W_0` and `W_1` (with Kaiming initialization), plus the covarite `cov[i]`. It is ultimately either `W_0` or `W_0+W_1` depending on whether cov[i] = 0 or 1
  - As discussed, cov is an array of covariates (male=1, female=0) and the i-th cov is a scalar. 
  - In the forward pass, we have a for loop that iterates over each data point in the mini batch
  - for each data point, kernel is computed as k_i = W_0 + W_1 * S_i where it is a scalar multiplication. 
  - Then we expand the first dimension of the image to make it has shape (1, 3, 224, 224)
  - Then a standard 2D convolution is done with the kernel k_i and a single image x[i]
  - Then after processing each data point in the minibatch, all the outputs of the Conv2d are concatenated into one
  - This output is the final output of the hybrid layer and then it becomes the input of the BatchNorm2d layer. 

Experiments:

The first time I ran the hybrid vgg it only gave me a 0.7 validation accuracy / f1 micro score. Then I ran 9 experiments with dummy covariates being all 0's, all 1's and the normal covariates in the hybrid layer, each crossed with batchsize = 16, 32, and 64. The results were all similar ranging from 0.78-0.80. 

I also noticed some very nice looking loss plots which has never occured before. That is, there was actually a visible downward loss curve instead of loss values just oscillating throughout the training.


TODO:
1. Cross validation using the train-val split; use the test set eventually
2. Make some graphics for your paper presentation; such as the hyrbrid CNN structure
3. Tune learning rate
4. One more feature than gender? 
5. Have a slideshow presentation

Manuscript Outline:
 - Introduction
 - Background (multi-modal ML - learning from structured info)
 - Literature Review (shortfuse, LiuNet, Flare)
 - CelebA dataset (target labels, pytorch support, train-val-test splits)
 - Network structure (graphics, hybrid conv)
 - Experiments (tables, baseline, hybrid, Batch input, different LR)
 - Observation (eevaluation metrics + results + plots)
 - Discussion / Future direction (on 3D medical imaging)


 Note: the 1e-4 in results are actually 10e-4. etc