import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils import FaceDetector

# Configuración
MODEL_PATH = "models/emotion_detection.h5"
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Cargar modelo y detector
model = load_model(MODEL_PATH)
detector = FaceDetector(detector_type='haar')  # Cambiar a 'mtcnn' para mayor precisión

def preprocess_face(face_roi):
    """Preprocesa un rostro para el modelo."""
    face_roi = cv2.resize(face_roi, (48, 48))
    face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    face_roi = face_roi.astype("float32") / 255.0
    return face_roi.reshape(1, 48, 48, 1)

# Captura de video
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    faces = detector.detect_faces(frame)
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        processed_face = preprocess_face(face_roi)
        preds = model.predict(processed_face)[0]
        emotion = EMOTIONS[np.argmax(preds)]
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame, f"{emotion}: {np.max(preds):.2f}",
            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2
        )
    
    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()