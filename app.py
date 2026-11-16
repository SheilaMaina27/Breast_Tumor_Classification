import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Breast Tumor Classification",
    page_icon="🔬",
    layout="centered"
)


# -----------------------------
# Load the trained model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "breast_tumor_mobilenetv2.keras"
    )


model = load_model()


# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess_image(image):
    # Convert image to grayscale
    image = image.convert("L")

    # Resize while preserving aspect ratio
    image = ImageOps.pad(
        image,
        (224, 224),
        method=Image.Resampling.LANCZOS,
        color=0
    )

    # Convert to NumPy array
    image_array = np.array(image)

    # Add channel dimension
    image_array = np.expand_dims(image_array, axis=-1)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# -----------------------------
# App title
# -----------------------------
st.title("🔬 Breast Tumor Classification")

st.write(
    "Upload a breast tumor image to classify it as "
    "benign or malignant."
)


# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)


# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(uploaded_file)

    # Display image
    st.image(
        image,
        caption="Uploaded Image",
        width=400
    )

    # Preprocess
    processed_image = preprocess_image(image)

    # Make prediction
    probability = model.predict(processed_image)[0][0]

    # Classification
    if probability >= 0.5:
        prediction = "Malignant"
    else:
        prediction = "Benign"

    # Display result
    st.subheader("Prediction")

    if prediction == "Malignant":
        st.error("Prediction: Malignant")
    else:
        st.success("Prediction: Benign")

    # Display probability
    st.write(
        f"Malignant probability: {probability:.2%}"
    )