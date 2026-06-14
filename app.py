import streamlit as st
import numpy as np
import os
import requests
#from load_model import get_model
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense

MODEL_URL = "https://huggingface.co/asadalam/brain_mri_3dresnet_final-bucket"
MODEL_PATH = "brain_mri_3dresnet_final.keras"

def get_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")

        response = requests.get(MODEL_URL)

        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)

        print("Download complete.")

    return load_model(MODEL_PATH)



# ===== FIX FOR quantization_config ERROR =====

original_from_config = Dense.from_config

@classmethod
def fixed_from_config(cls, config):
    config.pop("quantization_config", None)
    return original_from_config(config)

Dense.from_config = fixed_from_config

# ===== LOAD MODEL =====

@st.cache_resource
def load_my_model():
    return get_model()

model = load_my_model()

st.success("Model Loaded Successfully!")
# ===== 4 CLASS NAMES =====

class_names = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented"
]

# ===== UI =====

st.title("Alzheimer MRI Detection")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

# ===== PREDICTION =====

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(
        img,
        caption="Uploaded MRI Image",
        use_container_width=True
    )

    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = np.max(prediction) * 100

    st.subheader("Prediction Result")

    st.success(f"Prediction: {predicted_class}")

    st.write(f"Confidence: {confidence:.2f}%")

    st.subheader("Class Probabilities")

    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]} : {prob*100:.2f}%")
