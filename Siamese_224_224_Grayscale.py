import numpy as np
import streamlit as st
import tempfile
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

# Define cosine distance function
def cosine_distance(vectors):
    x, y = vectors
    x = K.l2_normalize(x, axis=1)
    y = K.l2_normalize(y, axis=1)
    return -K.sum(x * y, axis=1, keepdims=True)

# Function to preprocess uploaded images
def preprocess_image(uploaded_file, target_size=(224, 224)):
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        img = load_img(temp_path, target_size=target_size, color_mode='grayscale')
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        return img
    return None

# Load model only if uploaded
st.title("Siamese Network Image Similarity")

uploaded_model = st.file_uploader("Upload Trained Model (.h5)", type=["h5"])

if uploaded_model:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_model:
        temp_model.write(uploaded_model.read())
        model_path = temp_model.name

    siamese_model = load_model(model_path, custom_objects={'cosine_distance': cosine_distance})
    st.success("Model loaded successfully!")

# Upload images
img1 = st.file_uploader("Upload Image 1", type=["jpg", "png", "jpeg"])
img2 = st.file_uploader("Upload Image 2", type=["jpg", "png", "jpeg"])

# Predict similarity when the button is clicked
if st.button("Start Processing"):
    if "siamese_model" not in locals():
        st.error("Please upload a model before processing.")
    else:
        processed_img1 = preprocess_image(img1)
        processed_img2 = preprocess_image(img2)

        if processed_img1 is None or processed_img2 is None:
            st.error("Please upload both images before processing.")
        else:
            prediction = siamese_model.predict([processed_img1, processed_img2])
            similarity = prediction[0][0]

            st.write("Similarity Score:", similarity)
            st.write(f"Prediction: {'Dissimilar' if similarity > 0.5 else 'Similar'}")
