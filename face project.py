import streamlit as st
import face_recognition
import os
import cv2
from PIL import Image
import numpy as np

# Charger les visages connus
known_faces_dir = "known_faces"
known_encodings = []
known_names = []

for filename in os.listdir(known_faces_dir):
    image = face_recognition.load_image_file(f"{known_faces_dir}/{filename}")
    encoding = face_recognition.face_encodings(image)[0]
    known_encodings.append(encoding)
    known_names.append(os.path.splitext(filename)[0])

# Interface
st.title("🔍 Reconnaissance Faciale en Temps Réel")
uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image testée", use_column_width=True)

    # Traitement
    image_np = np.array(image)
    rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Inconnu"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        cv2.rectangle(image_np, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image_np, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    st.image(image_np, caption="Résultat", use_column_width=True)

