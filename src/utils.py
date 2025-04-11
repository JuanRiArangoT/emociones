import os
import cv2
import numpy as np
from mtcnn import MTCNN  # Opcional

def load_fer_dataset(data_path, img_size=(48, 48)):
    """Carga imágenes desde carpetas organizadas por emociones."""
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    X, y = [], []
    
    for label, emotion in enumerate(emotions):
        emotion_path = os.path.join(data_path, emotion)
        for img_file in os.listdir(emotion_path):
            img = cv2.imread(os.path.join(emotion_path, img_file), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, img_size)
            X.append(img)
            y.append(label)
    
    return np.array(X), np.array(y)

class FaceDetector:
    """Wrapper para detectores de rostros (Haar o MTCNN)."""
    def __init__(self, detector_type='haar'):
        self.detector_type = detector_type
        if detector_type == 'haar':
            self.detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        else:  # mtcnn
            self.detector = MTCNN()
    
    def detect_faces(self, frame):
        if self.detector_type == 'haar':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, 1.3, 5)
            return [(x, y, w, h) for (x, y, w, h) in faces]
        else:
            results = self.detector.detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return [face['box'] for face in results]