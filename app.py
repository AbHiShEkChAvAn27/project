import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F

# Load the model
model = torch.load('/workspaces/project/DetectingPlantDiseases.pth', map_location=torch.device('cpu'))
model.eval()

# Define image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization for pre-trained models
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Define a prediction function
def predict(image):
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Streamlit App
st.title('Plant Disease Detection')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    image = preprocess_image(image)
    
    # Make prediction
    prediction = predict(image)

    # Output the prediction
    st.write(f'Predicted class: {prediction}')
