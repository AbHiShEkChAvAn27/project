import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F

def accuracy(outputs, labels):    
    _, preds = torch.max(outputs, dim=1)    
    correct = torch.sum(preds == labels).item()
    accuracy = correct / len(labels)    
    return torch.tensor(accuracy)

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

def ConvBlock(in_channels, out_channels, pool=False):        
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # 3x3 Convolution with padding
        nn.BatchNorm2d(out_channels),  # Batch Normalization to stabilize learning
        nn.ReLU(inplace=True)  # ReLU Activation (in-place for memory efficiency)
    ]    
    if pool:
        layers.append(nn.MaxPool2d(4))  # 4x4 Max Pooling to reduce spatial dimensions    
    return nn.Sequential(*layers)

class CNN_NeuralNet(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()                
        self.conv1 = ConvBlock(in_channels, 64)  # First convolution block
        self.conv2 = ConvBlock(64, 128, pool=True)  # Second convolution block with pooling        
        self.res1 = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 128)
        )                
        self.conv3 = ConvBlock(128, 256, pool=True)  # Third convolution block with pooling
        self.conv4 = ConvBlock(256, 512, pool=True)  # Fourth convolution block with pooling                
        self.res2 = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 512)
        )                
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),  # Max pooling to reduce feature map size
            nn.Flatten(),     # Flatten the feature map for the linear layer
            nn.Linear(512, num_diseases)  # Fully connected layer for classification
        )
    def forward(self, x):      
        out = self.conv1(x)
        out = self.conv2(out)                
        out = self.res1(out) + out        
        out = self.conv3(out)
        out = self.conv4(out)            
        out = self.res2(out) + out                
        out = self.classifier(out)
        return out

model = torch.load('/workspaces/project/DetectingPlantDiseases.pth', map_location=torch.device('cpu'))
model.eval()

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization for pre-trained models
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def predict(image):
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

st.title('Plant Disease Detection')
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)    
    image = preprocess_image(image)    
    prediction = predict(image)    
    st.write(f'Predicted class: {prediction}')
