import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
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
from deep_translator import GoogleTranslator

# Translator initialization
def translate_text(text, target_language):
    translation = GoogleTranslator(source='auto', target=target_language).translate(text)
    return translation

languages = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Tamil": "ta",
    "Telugu": "te",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa"
}

selected_language = st.sidebar.selectbox("Select Language", languages.keys())
language_code = languages[selected_language]
genai.configure(api_key="abc")

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

model = CNN_NeuralNet(in_channels=3, num_diseases=38)
model.load_state_dict(torch.load('dpd.pth', map_location=torch.device('cpu')))
model.eval()  

# Define the transformation for images (same as used during training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def predict_image(img, model):
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    xb = img_tensor.to(torch.device('cpu'))   # Send image to CPU if using CPU
    yb = model(xb)  # Get model predictions
    _, preds = torch.max(yb, dim=1)  # Get the predicted class
    return preds[0].item()  # Return the predicted class index


disease_classes =['Tomato___Late_blight', 'Tomato___healthy', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Soybean___healthy',
 'Squash___Powdery_mildew',  'Potato___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Tomato___Early_blight', 'Tomato___Septoria_leaf_spot',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Strawberry___Leaf_scorch', 'Peach___healthy', 'Apple___Apple_scab', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Bacterial_spot', 'Apple___Black_rot', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Peach___Bacterial_spot',
 'Apple___Cedar_apple_rust', 'Tomato___Target_Spot', 'Pepper,_bell___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Potato___Late_blight',
 'Tomato___Tomato_mosaic_virus', 'Strawberry___healthy', 'Apple___healthy', 'Grape___Black_rot', 'Potato___Early_blight', 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Common_rust_', 'Grape___Esca_(Black_Measles)', 'Raspberry___healthy', 'Tomato___Leaf_Mold', 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Pepper,_bell___Bacterial_spot', 'Corn_(maize)___healthy']

def get_location():
    g = geocoder.ip('me')
    if not g.latlng: 
        st.warning("Geolocation failed, using default location (Pimpri-Chinchwad).")
        return [18.6298, 73.7997]
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

st.title(translate_text('🌿 Plant Disease Detection and Weather Info', language_code))
api_key = 'abc'  
weather = get_weather(api_key)

if weather:
    st.subheader(translate_text("☀️ Current Weather Information",language_code))
    st.write(translate_text(f"**Location:** {weather['name']}",language_code))
    st.write(translate_text(f"**Temperature:** {weather['main']['temp']}°C",language_code))
    st.write(translate_text(f"**Weather:** {weather['weather'][0]['description'].capitalize()}",language_code))

st.subheader(translate_text('🖼️ Upload Plant Image for Disease Detection', language_code))
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
    st.subheader(translate_text(f'🌿 Predicted Disease: {disease_classes[prediction]}',language_code))
    
    if weather:
        oresponse = get_response(prediction, we,wdse)
        st.subheader(translate_text("🌦️ Effect of Weather on Disease and Prevention",language_code))
        st.write(translate_text(oresponse,language_code))
