import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from utils import FaceDetector
from compare_models import show_model_comparison, show_model_metrics

# Configuración
MODEL_PATH = "models/best_model.h5"  # Usar el symlink creado por train.py
EMOTIONS = ['enojado', 'disgusto', 'miedo', 'feliz', 'neutral', 'triste', 'sorpresa']

# Título de la app
st.title("Detección de Emociones en Tiempo Real 🎭")
st.markdown("---")

# Nueva sección para comparación de modelos
if st.sidebar.checkbox("Mostrar análisis de modelos", False):
    st.header("Comparación de Modelos Entrenados")
    show_model_comparison()
    show_model_metrics()
    st.markdown("---")

# Sidebar para configuración
st.sidebar.header("Opciones")
use_webcam = st.sidebar.checkbox("Usar cámara web", True)
detector_type = st.sidebar.selectbox("Tipo de detector", ["haar", "mtcnn"], index=0)

# Cargar modelo con cache
@st.cache_resource
def load_emotion_model():
    return tf.keras.models.load_model(MODEL_PATH)

# Cargar detector de rostros
@st.cache_resource
def load_face_detector():
    return FaceDetector(detector_type=detector_type)

model = load_emotion_model()
detector = load_face_detector()

# Función para preprocesar el rostro
def preprocess_face(face_roi):
    face_roi = cv2.resize(face_roi, (48, 48))
    if model.input_shape[-1] == 3:  # Si el modelo espera RGB
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    else:  # Si espera escala de grises
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_roi = np.expand_dims(face_roi, -1)
    face_roi = face_roi.astype("float32") / 255.0
    return np.expand_dims(face_roi, 0)

# Interfaz principal
frame_placeholder = st.empty()
stop_button = st.button("Detener")

# Usar la nueva API de Streamlit para video
if use_webcam:
    video_capture = cv2.VideoCapture(0)
else:
    video_capture = cv2.VideoCapture("video.mp4")

while video_capture.isOpened() and not stop_button:
    ret, frame = video_capture.read()
    if not ret:
        st.error("Error al capturar video")
        break
    
    # Detección de rostros
    faces = detector.detect_faces(frame)
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        processed_face = preprocess_face(face_roi)
        preds = model.predict(processed_face, verbose=0)[0]
        emotion = EMOTIONS[np.argmax(preds)]
        confidence = np.max(preds)

        # Dibujar rectángulo y texto
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame, f"{emotion}: {confidence:.2f}",
            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2
        )
    
    # Mostrar frame en Streamlit
    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

video_capture.release()