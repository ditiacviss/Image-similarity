import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import streamlit as st
import tempfile

# Define custom Cosine Similarity Layer
class CosineSimilarityLayer(Layer):
    def __init__(self, **kwargs):
        super(CosineSimilarityLayer, self).__init__(**kwargs)

    def call(self, inputs):
        x, y = inputs
        x = K.l2_normalize(x, axis=1)
        y = K.l2_normalize(y, axis=1)
        return -K.sum(x * y, axis=1, keepdims=True)

    def get_config(self):
        return super().get_config()

# Function to load Siamese model from JSON & weights
def load_siamese_model(json_path, weights_path):
    with open(json_path, "r") as json_file:
        loaded_model_json = json_file.read()
    siamese_model = model_from_json(loaded_model_json, custom_objects={'CosineSimilarityLayer': CosineSimilarityLayer})

    # Load Weights
    siamese_model.load_weights(weights_path)
    print("Model loaded successfully!")
    return siamese_model

# Function to preprocess uploaded images
def preprocess_image(uploaded_file, target_size=(224, 224)):
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        img = load_img(temp_path, target_size=target_size, color_mode="grayscale")
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    return None

# Prediction function
def predict_similarity(model, img1, img2):
    if img1 is None or img2 is None:
        st.error("Please upload both images.")
        return None, None

    similarity_score = model.predict([img1, img2])[0][0]
    return 1 if similarity_score > 0.45 else 0, similarity_score

# Streamlit UI
st.title("Siamese Network Image Similarity")

json_path = "model/siamese_model.json"
weights_path = "model/siamese_model_weights.h5"

# Upload images
img1 = st.file_uploader("Upload Image 1", type=["jpg", "png", "jpeg"])
img2 = st.file_uploader("Upload Image 2", type=["jpg", "png", "jpeg"])


if st.button("Start Processing"):
    processed_img1 = preprocess_image(img1)
    processed_img2 = preprocess_image(img2)

    if processed_img1 is None or processed_img2 is None:
        st.error("Please upload both images before processing.")
    else:
        model = load_siamese_model(json_path, weights_path)
        predicted_label, similarity_score = predict_similarity(model, processed_img1, processed_img2)

        if predicted_label is not None:
            st.write("Similarity Score:", similarity_score)
            st.write(f"Prediction: {'Dissimilar' if predicted_label == 1 else 'Similar'}")
