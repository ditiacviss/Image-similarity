import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import streamlit as st
import tempfile
import os

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

def cosine_distance(vectors):
    x, y = vectors
    x = K.l2_normalize(x, axis=1)
    y = K.l2_normalize(y, axis=1)
    return -K.sum(x * y, axis=1, keepdims=True)

def predict(model, img1, img2):
    if img1 is None or img2 is None:
        st.error("Please upload both images.")
        return None

    prediction = model.predict([img1, img2])
    similarity = prediction[0][0]
    st.write('Similarity Score:', similarity)
    return 1 if similarity > 0.5 else 0

siamese_model = load_model(
    'model/siamese_model_19Feb.h5',
    custom_objects={'cosine_distance': cosine_distance}
)

img1 = st.file_uploader('Upload image 1')
img2 = st.file_uploader('Upload image 2')

if st.button('Start Processing'):
    processed_img1 = preprocess_image(img1)
    processed_img2 = preprocess_image(img2)
    predicted_label = predict(siamese_model,processed_img1,processed_img2)

    if predicted_label is not None:
        st.write(f"Prediction: {'Dissimilar' if predicted_label == 1 else 'Similar'}")
