import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import load_fer_dataset
from sklearn.utils import shuffle
import pickle
import os
from pathlib import Path
import shutil

# Configuración de rutas
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Configuración general
DATA_PATH = "data/raw/archive/train"
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 30

# Carga y preparación de datos
X_train, y_train = load_fer_dataset(DATA_PATH, IMG_SIZE)
X_train = X_train.reshape((-1, *IMG_SIZE, 1))
X_train, y_train = shuffle(X_train, y_train, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

# Modelo CNN desde cero
def build_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*IMG_SIZE, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(7, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Modelo con Transfer Learning
def build_transfer_model():
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(48, 48, 3)
    )
    base_model.trainable = False

    model = models.Sequential([
        layers.Rescaling(1./255),
        layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dense(7, activation="softmax")
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Función para comparar modelos
def compare_models(cnn_history, transfer_history):
    cnn_best_val_acc = max(cnn_history.history['val_accuracy'])
    transfer_best_val_acc = max(transfer_history.history['val_accuracy'])
    
    if cnn_best_val_acc > transfer_best_val_acc:
        best_model = "cnn"
        print(f"\n🔵 CNN es mejor (Val Accuracy: {cnn_best_val_acc:.4f} vs {transfer_best_val_acc:.4f})")
        # Copiar en lugar de symlink
        shutil.copy2(MODELS_DIR / "cnn_emotion_detection.h5", MODELS_DIR / "best_model.h5")
    else:
        best_model = "transfer"
        print(f"\n🟢 Transfer Learning es mejor (Val Accuracy: {transfer_best_val_acc:.4f} vs {cnn_best_val_acc:.4f})")
        # Copiar en lugar de symlink
        shutil.copy2(MODELS_DIR / "transfer_emotion_detection.h5", MODELS_DIR / "best_model.h5")
    
    # Guardar metadatos
    with open(MODELS_DIR / "best_model.txt", "w") as f:
        f.write(best_model)
    
    return best_model

# Entrenamiento y evaluación
def train_and_compare():
    # Entrenar CNN
    print("\nEntrenando CNN desde cero...")
    cnn_model = build_cnn_model()
    cnn_history = cnn_model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, subset='training'),
        validation_data=datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, subset='validation'),
        epochs=EPOCHS
    )
    cnn_model.save(MODELS_DIR / "cnn_emotion_detection.h5")

    # Entrenar Transfer Learning
    print("\nEntrenando modelo con Transfer Learning...")
    transfer_model = build_transfer_model()
    transfer_history = transfer_model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, subset='training'),
        validation_data=datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, subset='validation'),
        epochs=EPOCHS
    )
    transfer_model.save(MODELS_DIR / "transfer_emotion_detection.h5")

    # Evaluación adicional para guardar datos de métricas
    val_generator = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, subset='validation')
    X_val, y_val = next(val_generator)  # Obtener un batch de validación

    # Predicciones para ambos modelos
    cnn_preds = cnn_model.predict(X_val)
    transfer_preds = transfer_model.predict(X_val)

    # Guardar datos para métricas
    with open(MODELS_DIR / "evaluation_data.pkl", "wb") as f:
        pickle.dump({
            "y_true": y_val,
            "cnn_preds": cnn_preds,
            "transfer_preds": transfer_preds,
            "classes": ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        }, f)

    # Guardar historiales y comparar
    with open(MODELS_DIR / "training_history.pkl", "wb") as f:
        pickle.dump({
            "cnn_history": cnn_history.history,
            "transfer_history": transfer_history.history
        }, f)

    best_model = compare_models(cnn_history, transfer_history)
    
    # Crear symlink al mejor modelo
    best_model_path = MODELS_DIR / "best_model.h5"
    if best_model_path.exists():
        best_model_path.unlink()

    if best_model == "cnn":
        shutil.copy(MODELS_DIR / "cnn_emotion_detection.h5", best_model_path)
    else:
        shutil.copy(MODELS_DIR / "transfer_emotion_detection.h5", best_model_path)

if __name__ == "__main__":
    train_and_compare()