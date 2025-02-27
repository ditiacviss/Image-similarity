import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import os

# Define cosine distance function
def cosine_distance(vectors):
    x, y = vectors
    x = K.l2_normalize(x, axis=1)
    y = K.l2_normalize(y, axis=1)
    return -K.sum(x * y, axis=1, keepdims=True)

def preprocess_image(image_path, target_size=(224, 224)):
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")
        img = load_img(image_path, target_size=target_size, color_mode='grayscale')
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    except Exception as e:
        return None

def predict(model, imgage_pair):

    prediction = model.predict(imgage_pair)
    similarity = prediction[0][0]
    print(similarity)
    return 1 if similarity > 0.3 else 0  # 1 for dissimilar, 0 for similar


siamese_model = load_model(
    'model/siamese_model_19Feb.h5',
    custom_objects={'cosine_distance': cosine_distance}
)

img1_path = input('Enter the path for Image 1: ').strip()
img2_path = input('Enter the path for Image 2: ').strip()

img1_pre = preprocess_image(img1_path)
img2_pre = preprocess_image(img2_path)

image_pair = (img1_pre, img2_pre)

predicted_label = predict(siamese_model, image_pair)

if predicted_label is not None:
    print(f"Prediction: {'Dissimilar' if predicted_label == 1 else 'Similar'}")
