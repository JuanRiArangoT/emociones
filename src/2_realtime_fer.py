import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Configuración
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
model = load_model('../models/modelo_fer_opt.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Función para preprocesar rostro
def preprocess_face(face_roi):
    face_roi = cv2.resize(face_roi, (48, 48))
    face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    face_roi = face_roi.astype('float32') / 255.0
    face_roi = np.expand_dims(face_roi, axis=(0, -1))  # Shape: (1, 48, 48, 1)
    return face_roi

# Captura de video
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        processed_face = preprocess_face(face_roi)
        preds = model.predict(processed_face)[0]
        emotion = emotion_labels[np.argmax(preds)]
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame, f"{emotion}: {np.max(preds):.2f}",
            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2
        )
    
    cv2.imshow('FER2013 - Real Time Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()