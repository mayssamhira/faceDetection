import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("👁️ Face + Eyes + Mouth Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Lire image avec PIL
    pil_image = Image.open(uploaded_file).convert('RGB')
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    st.image(img, channels="BGR", caption="Original Image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Charger les cascades
    face_cascade = cv2.CascadeClassifier("pages/haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_mcs_mouth.xml")

    # Vérification
    if face_cascade.empty() or eye_cascade.empty() or mouth_cascade.empty():
        st.error("⚠ Un ou plusieurs fichiers Haar Cascade sont introuvables !")
    else:
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            face_gray = gray[y:y+h, x:x+w]
            face_color = img[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(face_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(face_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

            mouths = mouth_cascade.detectMultiScale(face_gray, 1.5, 11)
            for (mx, my, mw, mh) in mouths:
                if my > h / 2:  # bouche = moitié inférieure du visage
                    cv2.rectangle(face_color, (mx, my), (mx+mw, my+mh), (0, 0, 255), 2)
                    break

        st.image(img, channels="BGR", caption="Detected Features")
else:
    st.info("Upload an image to start.")
