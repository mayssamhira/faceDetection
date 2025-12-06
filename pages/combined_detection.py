import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("👁️ Combined Face Detection")

detection_type = st.selectbox("Choose Detection Type", ["Face Detection", "Face + Eyes Detection"])

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read image with PIL
    pil_image = Image.open(uploaded_file).convert('RGB')
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    st.image(img, channels="BGR", caption="Original Image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load cascades
    face_cascade = cv2.CascadeClassifier("pages/haarcascade_frontalface_default.xml")

    if detection_type == "Face Detection":
        if face_cascade.empty():
            st.error("⚠ Haar Cascade file not found!")
        else:
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            st.image(img, channels="BGR", caption="Detected Faces")

    elif detection_type == "Face + Eyes Detection":
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        if face_cascade.empty() or eye_cascade.empty():
            st.error("⚠ One or more Haar Cascade files not found!")
        else:
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                face_gray = gray[y:y+h, x:x+w]
                face_color = img[y:y+h, x:x+w]

                eyes = eye_cascade.detectMultiScale(face_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(face_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

            st.image(img, channels="BGR", caption="Detected Faces and Eyes")
else:
    st.info("Upload an image to start.")
