import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
from PIL import Image

# — Driver behavior class labels —
CLASS_NAMES = [
    "Other Activities",
    "Safe Driving",
    "Talking Phone",
    "Texting Phone",
    "Turning"
]

# — Load all models once —
@st.cache_resource
def load_models():
    driver_m = load_model(
        "driver_behaviour.keras",
        custom_objects={"preprocess_input": efficientnet_preprocess_input},
        compile=False
    )
    drow_m = load_model(
        "final_drowsiness_model.keras",
        custom_objects={"preprocess_input": mobilenet_preprocess_input},
        compile=False
    )
    steer_m = load_model("advanced_final_steering_model.keras", compile=False)
    return driver_m, drow_m, steer_m

driver_model, drowsiness_model, steering_model = load_models()

# — Streamlit App UI —
st.set_page_config(page_title="Smart Driver Monitor", layout="wide")
st.title("🚗 Smart Driver Monitoring Dashboard")

uploaded = st.file_uploader("📤 Upload a frame", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # — Common preprocessing function —
    def prep(image, size, preprocess_fn=None):
        image = image.resize(size)
        arr = np.array(image, dtype="float32")
        if preprocess_fn:
            arr = preprocess_fn(arr)
        arr = np.expand_dims(arr, 0)
        return arr

    # --- Driver Behavior Prediction ---
    beh_arr = prep(img, (224, 224), efficientnet_preprocess_input)
    beh_preds = driver_model.predict(beh_arr)[0]
    beh_idx = np.argmax(beh_preds)
    beh_label = CLASS_NAMES[beh_idx]
    beh_conf = beh_preds[beh_idx]

    # --- Drowsiness Detection ---
    drow_arr = prep(img, (224, 224), mobilenet_preprocess_input)
    drow_prob = float(drowsiness_model.predict(drow_arr)[0][0])
    alert_prob = 1 - drow_prob
    drowsy_status = "😴 Drowsy" if drow_prob > 0.5 else "🙂 Alert"

    # --- Steering Angle Prediction ---
    steer_arr = prep(img, (160, 80)) / 255.0
    steer_angle = float(steering_model.predict(steer_arr)[0][0])

    # — Layout: Three Columns —
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.subheader("🧠 Driver Behaviour")
        st.metric(label=beh_label, value=f"{beh_conf:.1%}")
        st.bar_chart(
            {name: float(p) for name, p in zip(CLASS_NAMES, beh_preds)},
            use_container_width=True
        )

    with col2:
        st.subheader("😴 Drowsiness Detection")
        if drow_prob > 0.5:
            st.error(f"**{drowsy_status}**", icon="⚠️")
        else:
            st.success(f"**{drowsy_status}**", icon="✅")

        st.metric(
            label="Alertness",
            value=f"{alert_prob:.1%}",
            delta=f"{drow_prob:.1%} drowsy",
            delta_color="inverse"
        )
        st.progress(drow_prob, text="Drowsiness Risk")

    with col3:
        st.subheader("🛞 Steering Angle")
        st.metric(label="Angle", value=f"{steer_angle:.2f}°")
        if abs(steer_angle) > 10:
            st.warning("⚠️ High steering angle!")
        else:
            st.success("✅ Within normal range")

    st.markdown("---")
    st.caption("⚙️ Powered by EfficientNet, MobileNetV2 & Custom CNNs")