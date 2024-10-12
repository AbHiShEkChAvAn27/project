import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms 
import requests
import geocoder
import google.generativeai as genai
import os

genai.configure(api_key="AIzaSyB4ZcIWZ0tiDwmgmuquF8HdzcIgmhugQQ8")

# Plant Disease Detection Setup
train_path="/workspaces/project/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
train = ImageFolder(train_path, transform=transforms.ToTensor())

def accuracy(outputs, labels):    
    _, preds = torch.max(outputs, dim=1)    
    correct = torch.sum(preds == labels).item()
    accuracy = correct / len(labels)    
    return torch.tensor(accuracy)

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):       
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(f"Epoch [{epoch}], train_loss: {result['train_loss']:.4f}, val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")

def ConvBlock(in_channels, out_channels, pool=False):        
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class CNN_NeuralNet(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 128)
        )
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 512)
        )
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_diseases)
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

# Load model
model = torch.load('/workspaces/project/DetectingPlantDiseases2.pth', map_location=torch.device('cpu'))
model.eval()

device = "cpu"
def to_device(data, device):    
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def preprocess_image(image):   
    transform = transforms.Compose([        
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def predict_image(image, model):    
    image = preprocess_image(image)
    xb = to_device(image, device)
    with torch.no_grad():
        yb = model(xb)
        probs = F.softmax(yb, dim=1)
        top_probs, top_classes = torch.topk(probs, k=2, dim=1)
    second_class = train.classes[top_classes.squeeze()[1].item()]
    return second_class

def get_location():
    g = geocoder.ip('me')
    if not g.latlng:  # Fallback if location is not found
        st.warning("Geolocation failed, using default location (Pimpri-Chinchwad).")
        return [18.6298, 73.7997]  # Pimpri-Chinchwad coordinates
    return g.latlng

def get_weather(api_key):
    lat, lon = get_location()
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        weather_data = response.json()
        return weather_data
    else:
        st.error("Failed to retrieve weather information.")
        return None

st.title('🌿 Plant Disease Detection and Weather Info')
api_key = 'xyz'  
weather = get_weather(api_key)

if weather:
    st.subheader("☀️ Current Weather Information")
    st.write(f"**Location:** {weather['name']}")
    st.write(f"**Temperature:** {weather['main']['temp']}°C")
    st.write(f"**Weather:** {weather['weather'][0]['description'].capitalize()}")

st.subheader("🖼️ Upload Plant Image for Disease Detection")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

def get_response(pre,we,wd):
    modelg = genai.GenerativeModel("gemini-1.5-flash")
    texts=f"what is effect of temperature {we} C and {wd} on the {pre} diesease for plant, and give its prevention and cure."
    response = modelg.generate_content(texts)
    return response.text

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    prediction = predict_image(image, model)
    we = weather['main']['temp']
    wdse = weather['weather'][0]['description']
    st.subheader(f'🌿 Predicted Disease: {prediction}')
    
    if weather:
        oresponse = get_response(prediction, we,wdse)
        st.subheader("🌦️ Effect of Weather on Disease and Prevention")
        st.write(oresponse)
