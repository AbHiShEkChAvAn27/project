import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil
# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import warnings
import time

from PIL import Image
import colorama
from colorama import Fore, Style
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader  # For loading and batching datasets
import torch.optim as optim
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
warnings.filterwarnings("ignore")
# %matplotlib inline

"""# **Preprocessing**"""

data_path = '/kaggle/input/new-plant-diseases-dataset/'

if os.path.exists(data_path):
    print(f"Found dataset directory: {data_path}")
    train_path = os.path.join(data_path, 'New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train')
    valid_path = os.path.join(data_path, 'New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid')
    test_path = os.path.join(data_path, 'test')
else:
    print(f"Error: Dataset directory not found at {data_path}")
    print("Please make sure the dataset is downloaded and mounted correctly.")

if os.path.exists(train_path):
    Diseases_Classes = os.listdir(train_path)
    print(f"Found {len(Diseases_Classes)} disease classes in the training set.")
else:
    print(f"Error: Train directory not found at {train_path}")
    print("Please check the path and ensure the directory exists.")

import os
import matplotlib.pyplot as plt

# Adjust these numbers as needed
rows = 5
cols = 5
num_classes_to_display = min(len(Diseases_Classes), rows * cols)

plt.figure(figsize=(40, 40), dpi=100)  # Adjust figure size and dpi for better performance
cnt = 0
plant_names = []
tot_images = 0

for i in Diseases_Classes[:num_classes_to_display]:
    cnt += 1
    plant_names.append(i)
    plt.subplot(rows, cols, cnt)

    image_path = os.listdir(os.path.join(train_path, i))
    print("The Number of Images in " + i + ":", len(image_path))
    tot_images += len(image_path)

    if image_path:  # Check if the list is not empty
        img_show = plt.imread(os.path.join(train_path, i, image_path[0]))
        plt.imshow(img_show)
    else:
        plt.text(0.5, 0.5, 'No Image', ha='center', va='center', fontsize=30)

    plt.xlabel(i, fontsize=30)
    plt.xticks([])
    plt.yticks([])

print("\nTotal Number of Images in Directory: ", tot_images)
plt.tight_layout()
plt.show()

plant_names = []
Len = []
for i in Diseases_Classes:
    plant_names.append(i)
    imgs_path = os.listdir(train_path + "/" + i)
    Len.append(len(imgs_path))

Len.sort(reverse=True)

sns.set(style="whitegrid", color_codes=True)
plt.figure(figsize=(20,20),dpi=200)
ax = sns.barplot(x= Len, y= plant_names, palette="Blues")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

train = ImageFolder(train_path, transform = transforms.ToTensor() )
valid = ImageFolder(valid_path, transform = transforms.ToTensor() )

train

train[100]

def show_image(image, label):
    print("Label :" + train.classes[label] + "(" + str(label) + ")")
    plt.imshow(image.permute(1, 2, 0))
image_list = [0, 3000, 5000, 8000, 12000, 15000, 60000, 70000]

chs = 0  # Counter to keep track of the subplot position
for img in image_list:
    chs += 1  # Increment the counter for each image
    plt.subplot(2, 4, chs)
    plt.tight_layout()
    plt.xlabel(img, fontsize=10)
    plt.title(train[img][1])
    show_image(*train[img])

# DataLoader for the training dataset
train_dataloader = DataLoader(
    train,            # The dataset for training (assumed to be defined earlier)
    batch_size=32,    # Number of samples per batch (32 samples will be processed together)
    shuffle=True,     # Shuffle the data at each epoch, which is important for training
    num_workers=2,    # Number of subprocesses used to load data (2 worker threads to load data in parallel)
    pin_memory=True   # If True, the data loader will copy tensors into CUDA pinned memory (faster transfers to GPU)
)

# DataLoader for the validation dataset
valid_dataloader = DataLoader(
    valid,            # The dataset for validation (assumed to be defined earlier)
    batch_size=32,    # Number of samples per batch (same size as TrainLoader for consistency)
    shuffle=True,     # Shuffle validation data; optional, but can ensure variety if the validation set is large
    num_workers=2,    # Use 2 worker threads to load data in parallel
    pin_memory=True   # Useful for GPU-based training; helps faster data transfer from host memory to GPU memory
)

"""## **1. Choosing the Device (get_default_device)**
## This function determines whether a GPU is available and returns the appropriate device (GPU or CPU).
"""

# for moving data into GPU (if available)
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

"""## **2. Moving Data to the Device (to_device)**
## This function moves a tensor (or a list of tensors) to the chosen device (CPU or GPU).
"""

# for moving data to device (CPU or GPU)
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

"""## **3. Automatically Moving Batches to the Device (DeviceDataLoader)**
## This class wraps a DataLoader and moves batches of data to the chosen device automatically.
"""

# for loading in the device (GPU if available else CPU)
class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dataloader:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dataloader)

device = get_default_device()
device

# Wrap the training DataLoader with DeviceDataLoader to automatically move batches to the specified device (GPU or CPU)
train_dataloader = DeviceDataLoader(train_dataloader, device)

# Wrap the validation DataLoader with DeviceDataLoader to automatically move batches to the specified device (GPU or CPU)
valid_dataloader = DeviceDataLoader(valid_dataloader, device)

# Function to calculate the accuracy of the model predictions
def accuracy(outputs, labels):
    # Get the predicted class by taking the index with the highest value (logit) along dimension 1
    _, preds = torch.max(outputs, dim=1)

    # Calculate the number of correct predictions and compute accuracy
    correct = torch.sum(preds == labels).item()
    accuracy = correct / len(labels)

    # Return the accuracy as a tensor
    return torch.tensor(accuracy)

# Base class for Image Classification models
class ImageClassificationBase(nn.Module):

    def training_step(self, batch):
        images, labels = batch  # Unpack the batch
        out = self(images)  # Generate predictions by passing images through the model
        loss = F.cross_entropy(out, labels)  # Calculate cross-entropy loss
        return loss

    def validation_step(self, batch):
        images, labels = batch  # Unpack the batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate cross-entropy loss
        acc = accuracy(out, labels)  # Calculate accuracy using the accuracy function
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]  # Extract loss from each batch
        epoch_loss = torch.stack(batch_losses).mean()  # Compute average loss over all batches
        batch_accs = [x['val_acc'] for x in outputs]  # Extract accuracy from each batch
        epoch_acc = torch.stack(batch_accs).mean()  # Compute average accuracy over all batches
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

# Function to create a convolutional block with optional pooling
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # 3x3 Convolution with padding
        nn.BatchNorm2d(out_channels),  # Batch Normalization to stabilize learning
        nn.ReLU(inplace=True)  # ReLU Activation (in-place for memory efficiency)
    ]

    # Add MaxPooling layer if pooling is enabled
    if pool:
        layers.append(nn.MaxPool2d(4))  # 4x4 Max Pooling to reduce spatial dimensions

    # Return the sequential block of layers
    return nn.Sequential(*layers)

"""# **Model Structure**

#### The **in_channels** parameter refers to the number of input channels for the convolutional layer in a Convolutional Neural Network (CNN) block like RGB Images havee 3 channels.
"""

# Custom ResNet-like CNN architecture for image classification
class CNN_NeuralNet(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()

        # Initial convolutional layers
        self.conv1 = ConvBlock(in_channels, 64)  # First convolution block
        self.conv2 = ConvBlock(64, 128, pool=True)  # Second convolution block with pooling

        # First residual block (two convolutional layers)
        self.res1 = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 128)
        )

        # Additional convolutional layers
        self.conv3 = ConvBlock(128, 256, pool=True)  # Third convolution block with pooling
        self.conv4 = ConvBlock(256, 512, pool=True)  # Fourth convolution block with pooling

        # Additional residual block (two convolutional layers)
        self.res2 = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 512)
        )

        # Classifier block: Pooling, Flatten, and Linear layer for classification
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),  # Max pooling to reduce feature map size
            nn.Flatten(),     # Flatten the feature map for the linear layer
            nn.Linear(512, num_diseases)  # Fully connected layer for classification
        )

    def forward(self, x):
        # Pass input through initial convolutional layers
        out = self.conv1(x)
        out = self.conv2(out)

        # Apply first residual block and add the input for skip connection
        out = self.res1(out) + out

        # Pass through additional convolutional layers
        out = self.conv3(out)
        out = self.conv4(out)

        # Apply second residual block and add the input for skip connection
        out = self.res2(out) + out

        # Pass through the classifier block to get final output
        out = self.classifier(out)
        return out

"""### **Why are we use numbers in layers like (16, 32, 64, 128, 256, 512)** ?

### Because of **Compatibility** with Hardware:

#### While you can use numbers like 25 or 89, it's often more efficient to use numbers that are powers of 2 (such as 32, 64, 128, etc.). This is because most hardware (like GPUs) is optimized for power-of-2 (2^n) memory usage and operations, making the training process faster and more memory-efficient.
"""

# Initialize the model with input channels and number of output classes
model = to_device(CNN_NeuralNet(3, len(train.classes)), device)

# Display the model architecture
model

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_OneCycle(epochs, max_lr, model, train_loader, val_loader, weight_decay=0,
                 grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()  # Clear GPU cache to free up memory
    history = []  # To store training results of each epoch

    # Initialize optimizer with weight decay (L2 regularization)
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)

    # Scheduler for One Cycle Learning Rate policy
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader)
    )

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        train_losses = []  # To track losses for the current epoch
        lrs = []  # To track learning rates for each batch

        for batch in train_loader:
            # Training step
            loss = model.training_step(batch)  # Compute loss
            train_losses.append(loss)  # Collect the loss for analysis
            loss.backward()  # Compute gradients

            # Gradient clipping (if specified)
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()  # Update model parameters
            optimizer.zero_grad()  # Reset gradients to zero for the next batch

            # Record the current learning rate
            lrs.append(get_lr(optimizer))
            sched.step()  # Update learning rate based on the One Cycle policy

        # Validation step at the end of each epoch
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()  # Average training loss
        result['lrs'] = lrs  # Learning rates used during the epoch
        model.epoch_end(epoch, result)  # Display epoch results
        history.append(result)  # Store results

    return history  # Return the history of training

history = [evaluate(model, valid_dataloader)]
history

num_epochs = 5                # Number of training epochs
learning_rate = 0.01          # Learning rate for the optimizer
grad_clip = 0.15              # Maximum gradient norm for gradient clipping
weight_decay = 1e-4           # Weight decay (L2 regularization) factor
optimizer_function = torch.optim.Adam  # Optimization function

history += fit_OneCycle(
    num_epochs,                 # Number of training epochs
    learning_rate,              # Maximum learning rate for the One Cycle policy
    model,                      # Neural network model to train
    train_dataloader,           # DataLoader for the training set
    valid_dataloader,           # DataLoader for the validation set
    grad_clip=grad_clip,        # Gradient clipping threshold
    weight_decay=weight_decay,  # Weight decay for regularization
    opt_func=optimizer_function # Optimization function (Adam)
)

val_acc = []
val_loss = []
train_loss = []

for i in history:
    val_acc.append(i['val_acc'])
    val_loss.append(i['val_loss'])
    train_loss.append(i.get('train_loss'))

import plotly.graph_objects as go
epoch_count = list(range(1, 7))
train_trace = go.Scatter(
    x=epoch_count,
    y=train_loss,
    mode='lines+markers',
    name='Training Loss',
    line=dict(color='orangered', dash='dash'),
    marker=dict(symbol='circle', size=8)
)

val_trace = go.Scatter(
    x=epoch_count,
    y=val_loss,
    mode='lines+markers',
    name='Validation Loss',
    line=dict(color='green', dash='dash'),
    marker=dict(symbol='circle', size=8)
)

fig = go.Figure(data=[train_trace, val_trace])
fig.update_layout(
    title='Training and Validation Loss Over Epochs',
    xaxis_title='Epoch',
    yaxis_title='Loss',
    legend_title='Loss Type',
    xaxis=dict(tickmode='linear'),
    template='plotly_white'
)
fig.show()

test = ImageFolder(test_path, transform=transforms.ToTensor())
test_images = sorted(os.listdir(test_path + '/test'))
print(test_images)
print(len(test_images))

def predict_image(img, model):
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return train.classes[preds[0].item()]

img, label = test[1]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[1], ', Predicted:', predict_image(img, model))

# predicting second image
img, label = test[20]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[26], ', Predicted:', predict_image(img, model))

# Save the entire model
torch.save(model, "Detecting Plant Diseases.pth")

# Commented out IPython magic to ensure Python compatibility.
# # prompt: create a streamlit web page for this model
# 
# %%writefile app.py
# import streamlit as st
# import torch
# from PIL import Image
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# 
# # Load the trained model
# model = torch.load("Detecting Plant Diseases.pth", map_location=torch.device('cpu'))
# model.eval()
# 
# # Define the image transformation pipeline
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])
# 
# def predict_image(img, model):
#     xb = transform(img).unsqueeze(0)
#     yb = model(xb)
#     _, preds  = torch.max(yb, dim=1)
#     return train.classes[preds[0].item()]
# 
# # Streamlit UI
# st.title("Plant Disease Detection")
# st.write("Upload an image of a plant to detect potential diseases.")
# 
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
# 
# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image.", use_column_width=True)
#     st.write("")
#     st.write("Classifying...")
# 
#     prediction = predict_image(image, model)
#     st.write("Prediction:", prediction)
# 
# 
#