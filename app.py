import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np


# Charger le modèle sauvegardé
model = load_model("brain_tumor_vgg_model.keras")

# Exemple de prédiction sur une image
def predict_image(model, img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return "Tumor" if prediction[0][0] > 0.5 else "No Tumor"

# Application Streamlit
st.title("Brain Tumor Classification")
st.write("Upload an image to predict whether it shows a brain tumor or not.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = "Tumor" if prediction[0][0] > 0.5 else "No Tumor"
    st.write(f"Prediction: {result}")

