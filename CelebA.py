from __future__ import print_function, division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import models, transforms, datasets

import torch.optim as optim
from torch.utils.data import DataLoader

import PIL.Image as Image
from tqdm import tqdm
import os
import time

from sklearn.metrics import f1_score

from model.vgg_16 import *
from model.hybrid_CNN import *



# get dataset
def load_data(batch_size):
    """
    return the train/val/test dataloader
    """
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])
    
    train_dataset = datasets.CelebA(root='./data',
                                    split='train',
                                    target_type='attr',
                                    transform=transform,
                                    download=False)
    val_dataset = datasets.CelebA(root='./data',
                                    split='valid',
                                    target_type='attr',
                                    transform=transform,
                                    download=False)
    test_dataset = datasets.CelebA(root='./data',
                                    split='test',
                                    target_type='attr',
                                    transform=transform,
                                    download=False)

    # data loader
    train_loader = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True)
    val_loader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    
    return train_loader, val_loader, test_loader


def initialize_model(model, learning_rate, num_classes, device):
    """
    initialize the model (pretrained vgg16_bn)
    define loss function and optimizer and move data to gpu if available
    
    return:
        model, loss function(criterion), optimizer
    """
    
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer

def make_plots(step_hist, loss_hist, epoch=0):
    plt.plot(step_hist, loss_hist)
    plt.xlabel('train_iterations')
    plt.ylabel('Loss')
    plt.title('epoch'+str(epoch+1))
    plt.savefig('plot1')
    plt.clf()


def train(train_loader, model, criterion, optimizer, num_epochs, device):
    """
    Move data to GPU memory and train for specified number of epochs
    Also plot the loss function and save it in `Figures/`
    Trained model is saved as `cnn.ckpt`
    """
    for epoch in range(num_epochs): # repeat the entire training `num_epochs` times
        # for each training sample
        loss_hist = []
        step_hist = []
        for i, (images, labels) in tqdm(enumerate(train_loader)):
            label = labels[:, 2]   # attractiveness label
            cov_attr = labels[:, 20]    # gender (male/female)   
            cov_attr = (cov_attr + 1) // 2  # map from {-1, 1} to {0, 1}
            
            # move to gpu if available
            images = images.to(device)
            label = label.to(device)
            
            # forward pass
            outputs = model(images, cov_attr)    # model takes covariate here
            loss = criterion(outputs, label) 
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # TODO: print AND log
            if (i+1) % 100 == 0:
                print('Epoch: [{}/{}], Step[{}/{}], Loss:{:.4f}' \
                        .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
                with open('output.txt', 'a') as f:
                    print('Epoch: [{}/{}], Step[{}/{}], Loss:{:.4f}' \
                        .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()), file=f)
                loss_hist.append(loss.item())
                step_hist.append(i+1)
        
        make_plots(step_hist, loss_hist, epoch)
        
    torch.save(model.state_dict(), 'cnn_with_cov.ckpt')


def evaluate(val_loader, model, device):
    """
    Run the validation set on the trained model
    """
    model.eval() # BatchNorm uses moving mean/variance instead of mini-batch mean/variance
    with torch.no_grad():
        # initialize the stats
        correct = 0
        total = 0
        # pass through testing data once
        for images, labels in val_loader:

            label = labels[:, 2]
            cov_attr = labels[:, 20]    # gender (male/female)   
            cov_attr = (cov_attr + 1) // 2  # map from {-1, 1} to {0, 1}
            
            # again move to device first
            images = images.to(device)
            label = label.to(device)
            
            # forward
            outputs = model(images, cov_attr)
            
            # instead of calculating loss we will get predictions
            # it's essetially outputs just reformatting imo
            _, predicted = torch.max(outputs.data, 1)
            # accumulate stats
            total += label.size(0) # yeah again, number of elements in the tensor
            correct += (label == predicted).sum().item()
        print('Validation accuracy: {}%'.format(100 * correct / total))
        with open('output.txt', 'a') as f:
            print('Validation accuracy: {}%'.format(100 * correct / total), file=f)
    
    
def main():
    # hyper parameters    
    num_epochs = 1
    num_classes = 2
    batch_size = 1 # WE WANT IMAGES TO PASS HYBRID CONV LAYER ONE BY ONE
    learning_rate = 0.001
    model_name = MyVGG16()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print("Loading data...")
    train_loader, val_loader, test_loader = load_data(batch_size)
    
    print("Initializing model...")
    model, criterion, optimizer = initialize_model(model_name, learning_rate, num_classes, device)
   
    print("Start training... \n")
    train(train_loader, model, criterion, optimizer, num_epochs, device)
    
    print("Start evaluating... \n")
    evaluate(val_loader, model, device)    

if __name__ == "__main__":
    main()
