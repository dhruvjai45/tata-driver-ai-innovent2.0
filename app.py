import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

# Set page config
st.set_page_config(page_title="Smart Driver Monitoring", layout="wide")

# Inject custom CSS for dark theme
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: #FAFAFA;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .uploadbox, .outputbox {
        border: 1px solid #444;
        background-color: #1c1c1c;
        border-radius: 10px;
        padding: 2rem;
        height: 100%;
    }
    .upload-area {
        border: 2px dashed #555;
        padding: 2rem;
        text-align: center;
        color: #888;
        cursor: pointer;
        border-radius: 10px;
        font-size: 1.2rem;
    }
    .stButton > button {
        background-color: #F39C12;
        color: black;
    }
    .stRadio > div {
        color: #FAFAFA;
    }
    </style>
""", unsafe_allow_html=True)

# Title and Info
st.title("🧠 Smart Driver Monitoring System")
st.caption("Detects driver drowsiness, behavior, and steering direction")

# Sidebar Model Choice
model_choice = st.sidebar.radio(
    "Select Model:",
    ["Drowsiness Detection", "Driver Behavior", "Steering Prediction"]
)

# Load Models
@st.cache_resource
def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

model_paths = {
    "Drowsiness Detection": "models/final_drowsiness_model.pt",
    "Driver Behavior": "models/driver_behaviour.pt",
    "Steering Prediction": "models/advanced_final_steering_model.pt"
}

model = load_model(model_paths[model_choice])

# Preprocessing
def preprocess_image(image, model_type):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = image.convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

# Class Labels
labels = {
    "Drowsiness Detection": ["Alert", "Drowsy"],
    "Driver Behavior": ["Normal", "Phone Call", "Drinking", "Smoking", "Distracted"],
    "Steering Prediction": ["Left", "Straight", "Right"]
}

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### Input Image")
    st.markdown('<div class="uploadbox">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if not uploaded_file:
        st.markdown('<div class="upload-area">Drop Image Here<br>– or –<br>Click to Upload</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("#### Prediction Output")
    st.markdown('<div class="outputbox">', unsafe_allow_html=True)

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with torch.no_grad():
            input_tensor = preprocess_image(image, model_choice)
            output = model(input_tensor)
            if model_choice == "Steering Prediction":
                prediction = torch.argmax(output, dim=1).item()
            else:
                prediction = torch.argmax(torch.softmax(output, dim=1), dim=1).item()

        st.success(f"🧾 Predicted: **{labels[model_choice][prediction]}**")
    else:
        st.markdown('<div class="upload-area">Waiting for Image...</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)