import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set page config
st.set_page_config(page_title="Driver Monitoring System", layout="centered")

# Title
st.title("🚗 Driver Monitoring Inference System")
st.markdown("""
Upload an image of a driver to detect:
- **Steering Prediction**
- **Drowsiness Detection**
- **Driver Behavior Classification**
""")

# Load models (make sure these paths are correct relative to app.py)
@st.cache_resource
def load_models():
    steering_model = tf.keras.models.load_model("models/advanced_final_steering_model.keras")
    drowsiness_model = tf.keras.models.load_model("models/final_drowsiness_model.keras")
    behavior_model = tf.keras.models.load_model("models/driver_behaviour.keras")
    return steering_model, drowsiness_model, behavior_model

steering_model, drowsiness_model, behavior_model = load_models()

# Optional: labels for behavior model
behavior_labels = ["Normal", "Texting", "Drinking", "Phone Call", "Other"]  # change as per your model

# Image preprocessing function
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# File uploader
uploaded_file = st.file_uploader("Upload Driver Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process and predict
    with st.spinner("Analyzing image..."):

        # Preprocess for each model (adjust target_size if needed)
        input_steering = preprocess_image(image, (224, 224))
        input_drowsiness = preprocess_image(image, (224, 224))
        input_behavior = preprocess_image(image, (224, 224))

        # Run predictions
        steering_pred = steering_model.predict(input_steering)
        drowsiness_pred = drowsiness_model.predict(input_drowsiness)
        behavior_pred = behavior_model.predict(input_behavior)

        # Format predictions (adjust based on your model output)
        # Example: regression output
        if steering_pred.shape[1] == 1:
            steering_result = f"Steering Angle: {steering_pred[0][0]:.2f}°"
        else:
            steering_result = f"Steering Class: {np.argmax(steering_pred)}"

        drowsiness_result = "Drowsy 😴" if drowsiness_pred[0][0] > 0.5 else "Alert 🙂"
        behavior_result = behavior_labels[np.argmax(behavior_pred)] if len(behavior_pred[0]) == len(behavior_labels) else f"Behavior class: {np.argmax(behavior_pred)}"

    # Display results
    st.success("✅ Predictions Complete:")
    st.markdown(f"**🧭 Steering Prediction:** {steering_result}")
    st.markdown(f"**😵 Drowsiness Detection:** {drowsiness_result}")
    st.markdown(f"**🧍 Driver Behavior:** {behavior_result}")
