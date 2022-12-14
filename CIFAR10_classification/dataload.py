# Upload data for training and save predictions

import os
import pickle
import pandas as pd
import numpy as np
from scipy.io import loadmat
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms

import settings

def trans(which):
    trans_types = ['train', 'test']
    if which not in trans_types:
        raise ValueError(f"Invalid transform type. Expected one of: {trans_types}")
    elif which == "train":
        train_transforms = transforms.Compose(
            [
                transforms.RandomRotation(degrees=(0, 180)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        )
        return train_transforms
    elif which == "test":
        test_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        )
        return test_transforms

def load_images():
    # Collecting images for training the model
    print("Collecting data")
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=trans('train'))
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=trans('test'))
    
    classes_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    num_classes = len(classes_names)
    
    return classes_names, trainset, testset
