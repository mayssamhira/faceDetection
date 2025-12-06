import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

st.title("🧑‍🦰 Face Detection")

uploaded_file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])

cascade_path = "pages/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    st.error("Erreur : le fichier Haar Cascade n'a pas été chargé !")

if uploaded_file:
    # Ouvrir l'image avec PIL et convertir en array numpy
    pil_image = Image.open(uploaded_file).convert('RGB')
    img = np.array(pil_image)
    # Convertir RGB -> BGR pour OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    st.image(img, channels="BGR")