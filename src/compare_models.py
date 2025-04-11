import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import streamlit as st

def plot_confusion_matrix(y_true, y_pred, classes):
    """Muestra matriz de confusión normalizada"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title("Matriz de Confusión (Normalizada)")
    plt.ylabel("Verdaderos")
    plt.xlabel("Predichos")
    st.pyplot(plt.gcf())
    plt.clf()

def plot_roc_curve(y_true, y_scores, classes):
    """Muestra curva ROC multiclase (One-vs-Rest)"""
    plt.figure(figsize=(10, 8))
    
    for i, emotion in enumerate(classes):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{emotion} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC por Emoción')
    plt.legend(loc="lower right")
    st.pyplot(plt.gcf())
    plt.clf()

def show_model_metrics():
    """Muestra todas las métricas del mejor modelo"""
    try:
        with open("models/evaluation_data.pkl", "rb") as f:
            eval_data = pickle.load(f)
        
        with open("models/best_model.txt", "r") as f:
            best_model = f.read().strip()
        
        # Seleccionar predicciones del mejor modelo
        y_scores = eval_data["transfer_preds"] if best_model == "transfer" else eval_data["cnn_preds"]
        y_pred = y_scores.argmax(axis=1)
        
        st.subheader("📊 Métricas del Mejor Modelo")
        plot_confusion_matrix(eval_data["y_true"], y_pred, eval_data["classes"])
        plot_roc_curve(eval_data["y_true"], y_scores, eval_data["classes"])
        
    except FileNotFoundError:
        st.error("Datos de evaluación no encontrados. Ejecuta train.py primero.")

def show_model_comparison():
    MODELS_DIR = Path("models")
    
    try:
        with open(MODELS_DIR / "training_history.pkl", "rb") as f:
            history = pickle.load(f)
        
        with open(MODELS_DIR / "best_model.txt", "r") as f:
            best_model = f.read().strip()
    except FileNotFoundError as e:
        st.error(f"Error: {e}")
        st.warning("Ejecuta train.py primero para generar los datos de comparación")
        return
    
    cnn_history = history["cnn_history"]
    transfer_history = history["transfer_history"]
    
    # Crear figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico de Accuracy
    ax1.plot(cnn_history['val_accuracy'], label='CNN desde cero', color='blue' if best_model == 'cnn' else 'gray')
    ax1.plot(transfer_history['val_accuracy'], label='Transfer Learning', color='green' if best_model == 'transfer' else 'gray')
    ax1.set_title('Comparación de Accuracy en Validación')
    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Gráfico de Loss
    ax2.plot(cnn_history['val_loss'], label='CNN desde cero', color='blue' if best_model == 'cnn' else 'gray')
    ax2.plot(transfer_history['val_loss'], label='Transfer Learning', color='green' if best_model == 'transfer' else 'gray')
    ax2.set_title('Comparación de Pérdida en Validación')
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    
    # Mostrar en Streamlit
    st.pyplot(fig)
    st.info(f"Mejor modelo seleccionado: {'CNN' if best_model == 'cnn' else 'Transfer Learning'}")