"""
Pipeline de Entrenamiento Avanzado con SMOTE + Tomek Links
Maximiza recall balanceando datos y limpiando frontera de decisión
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import class_weight
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
import matplotlib.pyplot as plt
import joblib

# Importar modelos mejorados
from IDSModelCNN_v2 import IDSModelCNN_v2, FocalLoss
from IDSModelLSTM_v2 import IDSModelLSTM_v2, AttentionLayer


def load_and_preprocess_data(train_path, test_path):
    """
    Carga datos sin balancear (SMOTE se aplica después)
    """
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
    ]
    
    train_df = pd.read_csv(train_path, names=columns)
    test_df = pd.read_csv(test_path, names=columns)
    
    full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    # Convertir label a binario
    full_df["label"] = full_df["label"].apply(lambda x: 0 if x == "normal" else 1)
    full_df = full_df.drop(['difficulty'], axis=1)
    
    # Encoding categóricas
    categorical_cols = ['protocol_type', 'service', 'flag']
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        full_df[col] = le.fit_transform(full_df[col])
        label_encoders[col] = le
    
    # Separar features y labels
    X = full_df.drop(['label'], axis=1).values
    y = full_df['label'].values
    
    # Dividir train/test
    train_size = len(train_df)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, label_encoders


def apply_smote_tomek(X_train, y_train, sampling_strategy='auto'):
    """
    Aplica SMOTE + Tomek Links
    - SMOTE: Genera ejemplos sintéticos de clase minoritaria
    - Tomek Links: Limpia frontera eliminando pares ambiguos
    """
    print(f"\n{'='*60}")
    print("APLICANDO SMOTE + TOMEK LINKS")
    print(f"{'='*60}")
    
    print(f"\nDistribución ANTES del balanceo:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Clase {label}: {count} ({count/len(y_train)*100:.2f}%)")
    
    # Detectar clase minoritaria y calcular ratio seguro
    class_counts = dict(zip(unique, counts))
    minority_class = min(class_counts, key=class_counts.get)
    majority_class = max(class_counts, key=class_counts.get)
    
    current_ratio = class_counts[minority_class] / class_counts[majority_class]
    
    # Si ya está bien balanceado (>0.7), usar 'auto' o ratio conservador
    if current_ratio > 0.7:
        print(f"\nClases ya bien balanceadas (ratio: {current_ratio:.2f})")
        print("Usando sampling_strategy='auto' (balanceo 1:1)")
        final_sampling_strategy = 'auto'
    else:
        # Si está desbalanceado, usar el ratio especificado
        final_sampling_strategy = sampling_strategy if isinstance(sampling_strategy, str) else min(sampling_strategy, 0.95)
    
    # Aplicar SMOTE + Tomek
    smote_tomek = SMOTETomek(
        smote=SMOTE(sampling_strategy=final_sampling_strategy, random_state=42, k_neighbors=5),
        tomek=TomekLinks(sampling_strategy='auto'),
        random_state=42
    )
    
    X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
    
    print(f"\nDistribución DESPUÉS del balanceo:")
    unique, counts = np.unique(y_resampled, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Clase {label}: {count} ({count/len(y_resampled)*100:.2f}%)")
    
    print(f"\nEjemplos añadidos: {len(X_resampled) - len(X_train)}")
    print(f"Total de ejemplos: {len(X_resampled)}")
    
    return X_resampled, y_resampled


def train_cnn_with_smote(train_path, test_path, model_name='best_cnn_v2_smote'):
    """
    Entrena CNN v2 con SMOTE + Tomek Links
    """
    print("\n" + "="*70)
    print("ENTRENANDO CNN v2 CON SMOTE + TOMEK LINKS")
    print("="*70)
    
    # Cargar datos
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(train_path, test_path)
    
    # Normalizar ANTES de SMOTE
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Aplicar SMOTE + Tomek
    X_train_balanced, y_train_balanced = apply_smote_tomek(X_train, y_train, sampling_strategy='auto')
    
    # Dividir en train/val DESPUÉS de balancear
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_balanced, y_train_balanced,
        test_size=0.2,
        random_state=42,
        stratify=y_train_balanced
    )
    
    # Reshape para CNN
    X_train_split = np.expand_dims(X_train_split, axis=2)
    X_val = np.expand_dims(X_val, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    
    # Crear y entrenar modelo
    cnn_model = IDSModelCNN_v2(model_path=f'{model_name}.h5')
    cnn_model.scaler = scaler  # Usar el scaler ya entrenado
    
    print("\nConstruyendo modelo...")
    cnn_model.build_model(input_shape=(X_train_split.shape[1], 1))
    
    print("\nIniciando entrenamiento...")
    history = cnn_model.train(
        X_train_split, y_train_split,
        X_val, y_val,
        epochs=50,
        batch_size=256  # Batch más grande por más datos
    )
    
    # Guardar scaler
    cnn_model.save_scaler(f'scaler_{model_name}.pkl')
    
    # Evaluar
    print("\n" + "="*70)
    print("EVALUACIÓN EN TEST SET")
    print("="*70)
    
    for threshold in [0.7, 0.65, 0.6, 0.55, 0.5]:
        cnn_model.evaluate(X_test, y_test, threshold=threshold)
    
    return cnn_model, history


def train_lstm_with_smote(train_path, test_path, model_name='best_lstm_v2_smote'):
    """
    Entrena LSTM v2 con SMOTE + Tomek Links
    """
    print("\n" + "="*70)
    print("ENTRENANDO LSTM v2 CON SMOTE + TOMEK LINKS")
    print("="*70)
    
    # Cargar datos
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(train_path, test_path)
    
    # Normalizar ANTES de SMOTE
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Aplicar SMOTE + Tomek
    X_train_balanced, y_train_balanced = apply_smote_tomek(X_train, y_train, sampling_strategy='auto')
    
    # Dividir en train/val
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_balanced, y_train_balanced,
        test_size=0.2,
        random_state=42,
        stratify=y_train_balanced
    )
    
    # Reshape para LSTM
    X_train_split = np.expand_dims(X_train_split, axis=2)
    X_val = np.expand_dims(X_val, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    
    # Crear y entrenar modelo
    lstm_model = IDSModelLSTM_v2(model_path=f'{model_name}.h5')
    lstm_model.scaler = scaler
    
    print("\nConstruyendo modelo...")
    lstm_model.build_model(input_shape=(X_train_split.shape[1], 1))
    
    print("\nIniciando entrenamiento...")
    history = lstm_model.train(
        X_train_split, y_train_split,
        X_val, y_val,
        epochs=50,
        batch_size=256
    )
    
    # Guardar scaler
    lstm_model.save_scaler(f'scaler_{model_name}.pkl')
    
    # Evaluar
    print("\n" + "="*70)
    print("EVALUACIÓN EN TEST SET")
    print("="*70)
    
    for threshold in [0.7, 0.65, 0.6, 0.55, 0.5]:
        lstm_model.evaluate(X_test, y_test, threshold=threshold)
    
    return lstm_model, history


def plot_training_history(history, model_name):
    """
    Visualiza métricas de entrenamiento
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val')
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train')
    axes[0, 1].plot(history.history['val_loss'], label='Val')
    axes[0, 1].set_title('Loss (Focal Loss)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Recall
    axes[1, 0].plot(history.history['recall'], label='Train')
    axes[1, 0].plot(history.history['val_recall'], label='Val')
    axes[1, 0].set_title('Recall (Métrica Clave)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Precision
    axes[1, 1].plot(history.history['precision'], label='Train')
    axes[1, 1].plot(history.history['val_precision'], label='Val')
    axes[1, 1].set_title('Precision')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Precision')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png', dpi=300, bbox_inches='tight')
    print(f"\nGráfico guardado: {model_name}_training_history.png")
    plt.close()


if __name__ == "__main__":
    # Paths del dataset
    TRAIN_PATH = "C:/Users/Diego/Downloads/NSL_KDD-master/NSL_KDD-master/KDDTrain+.csv"
    TEST_PATH = "C:/Users/Diego/Downloads/NSL_KDD-master/NSL_KDD-master/KDDTest+.csv"
    
    print("="*70)
    print("PIPELINE DE ENTRENAMIENTO AVANZADO")
    print("SMOTE + Tomek Links + Focal Loss + Arquitecturas Mejoradas")
    print("="*70)
    
    # Entrenar CNN con SMOTE
    print("\n[1/2] Entrenando CNN...")
    cnn_model, cnn_history = train_cnn_with_smote(
        TRAIN_PATH, TEST_PATH,
        model_name='best_cnn_v2_smote'
    )
    plot_training_history(cnn_history, 'cnn_v2_smote')
    
    # Entrenar LSTM con SMOTE
    print("\n[2/2] Entrenando LSTM...")
    lstm_model, lstm_history = train_lstm_with_smote(
        TRAIN_PATH, TEST_PATH,
        model_name='best_lstm_v2_smote'
    )
    plot_training_history(lstm_history, 'lstm_v2_smote')
    
    print("\n" + "="*70)
    print("✅ ENTRENAMIENTO COMPLETO")
    print("="*70)
    print("\nModelos guardados:")
    print("  - best_cnn_v2_smote.h5")
    print("  - best_lstm_v2_smote.h5")
    print("  - scaler_best_cnn_v2_smote.pkl")
    print("  - scaler_best_lstm_v2_smote.pkl")
    print("\nGráficos:")
    print("  - cnn_v2_smote_training_history.png")
    print("  - lstm_v2_smote_training_history.png")
    print("\nPróximo paso: ejecutar ensemble_v2.py para combinar ambos modelos")
