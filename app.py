import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from detecting-plant-diseases-pytorch import CNN_NeuralNet


# Load the saved model
loaded_model=CNN_NeuralNet()
loaded_model = torch.load("/workspaces/project/DetectingPlantDiseases.pth", map_location=torch.device('cpu'))

# Function to predict image
def predict_image(img, model):        
    xb = img.unsqueeze(0)    
    yb = model(xb)    
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    predicted_class = train.classes[preds[0].item()]
    # Remove underscores from the predicted class
    predicted_class = predicted_class.replace("_", " ")
    return predicted_class

# Streamlit app
st.title("Plant Disease Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(image)

    # Make prediction
    prediction = predict_image(img_tensor, loaded_model)
    st.write("Prediction:", prediction)

