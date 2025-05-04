Python 3.13.3 (tags/v3.13.3:6280bb5, Apr  8 2025, 14:47:33) [MSC v.1943 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
import numpy as np
import cv2
from mtcnn import MTCNN
from keras.models import load_model
from sklearn.svm import SVC
from flask import Flask, request, jsonify
import os

# Chargement du modèle FaceNet
facenet_model = load_model("facenet_keras.h5")
detector = MTCNN()

def preprocess_image(image):
    image = cv2.resize(image, (160, 160))
    return (image.astype('float32') - 127.5) / 128.0

def extract_face(image):
    faces = detector.detect_faces(image)
    if not faces:
        return None
    x, y, w, h = faces[0]['box']
    face = image[y:y+h, x:x+w]
    return preprocess_image(face)

def get_embedding(face):
    face = np.expand_dims(face, axis=0)
    return facenet_model.predict(face)[0]
... 
... # Exemple de données fictives pour entraînement (à remplacer par de vraies images)
... embeddings = []
... labels = []
... 
... # Dossier d’images par personne : dataset/person1/*.jpg
... for person_name in os.listdir("dataset"):
...     person_path = os.path.join("dataset", person_name)
...     if not os.path.isdir(person_path):
...         continue
...     for filename in os.listdir(person_path):
...         path = os.path.join(person_path, filename)
...         image = cv2.imread(path)
...         face = extract_face(image)
...         if face is not None:
...             emb = get_embedding(face)
...             embeddings.append(emb)
...             labels.append(person_name)
... 
... # Entraînement du modèle
... classifier = SVC(kernel='linear', probability=True)
... classifier.fit(embeddings, labels)
... 
... # Flask API
... app = Flask(__name__)
... 
... @app.route('/predict', methods=['POST'])
... def predict():
...     file = request.files['image']
...     img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
...     face = extract_face(img)
...     if face is None:
...         return jsonify({'error': 'No face detected'})
...     emb = get_embedding(face)
...     prediction = classifier.predict([emb])[0]
...     return jsonify({'identity': prediction})
... 
... if __name__ == "__main__":
