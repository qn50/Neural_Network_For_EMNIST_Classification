# Neural_Network_For_EMNIST_Classification

Neural Network for EMNIST Classification
Overview
This repository contains code for training neural networks on the EMNIST (Extended Modified National Institute of Standards and Technology) dataset using PyTorch. The code demonstrates how to load the EMNIST dataset, preprocess the data, define neural network architectures with one and two layers, train the models, and evaluate their performance.

# Prerequisites
Before running the code, ensure you have the required libraries installed:
%pip install torch torchvision
%pip install matplotlib

Getting Started
To get started, run the provided Jupyter notebook. The notebook begins by installing the necessary libraries and then loads and preprocesses the EMNIST dataset using PyTorch and torchvision.
from IPython.display import clear_output

# Install required libraries
%pip install torch torchvision
%pip install matplotlib

clear_output()

# Import necessary libraries
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.datasets import EMNIST
import torchvision.transforms.functional as F

# Data Loading and Preprocessing
The code uses the EMNIST dataset, splits it into training and testing sets, and applies normalization transformations.
normalize_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_data = EMNIST(root='emnist_data/', split='byclass', download=True, transform=normalize_transform)
test_data = EMNIST(root='emnist_data/', split='byclass', download=True, train=False, transform=normalize_transform)

# Neural Network Architectures
Two neural network architectures are defined: NN1Layer with one layer and NN2Layer with two layers. The code also sets up the necessary configurations for training, such as the number of epochs, learning rate, and device (CPU or GPU).

# Training and Evaluation
The models are trained and evaluated using cross-entropy loss. The training loop includes both training and validation phases, and the performance metrics are printed for each epoch.

# Training loop
for epoch_no in range(num_epochs):
    # Training phase
    # ...

    # Validation phase
    # ...

    print(f'Epoch: {epoch_no}, train_loss={epoch_loss}, val_loss={val_epoch_loss}. '
          f'Labelled {correctly_labelled}/{len(test_loader.dataset)} correctly '
          f'({correctly_labelled/len(test_loader.dataset)*100}% accuracy)')

# Results Visualization
The training and validation losses are plotted over epochs to visualize the training progress.

plt.plot(train_losses, label='train loss')
plt.plot(val_losses, label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (Cross Entropy)')
plt.legend()
plt.show()
