"""
CNN Mejorado para IDS - Versión 2
Optimizado para maximizar recall en detección de ataques
Incluye: Focal Loss, arquitectura profunda, BatchNormalization, Dropout adaptativo
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import joblib


class FocalLoss(keras.losses.Loss):
    """
    Focal Loss para manejar desbalanceo de clases
    Reduce peso de ejemplos bien clasificados, enfoca en difíciles
    """
    def __init__(self, alpha=0.25, gamma=2.0, name='focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        # Clip predictions para evitar log(0)
        y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1 - keras.backend.epsilon())
        
        # Calcular cross entropy
        ce = -y_true * tf.math.log(y_pred)
        
        # Aplicar focal term
        focal_weight = tf.pow(1 - y_pred, self.gamma)
        focal_loss = self.alpha * focal_weight * ce
        
        return tf.reduce_mean(focal_loss)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class IDSModelCNN_v2:
    def __init__(self, model_path=None):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model_path = model_path
        
    def load_and_preprocess(self, train_path, test_path):
        """
        Carga y preprocesa datos con encoding mejorado
        """
        # Columnas del dataset NSL-KDD
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
        
        # Cargar datos
        train_df = pd.read_csv(train_path, names=columns)
        test_df = pd.read_csv(test_path, names=columns)
        
        # Combinar para encoding consistente
        full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
        
        # Convertir label a binario (0=normal, 1=attack)
        full_df["label"] = full_df["label"].apply(lambda x: 0 if x == "normal" else 1)
        
        # Eliminar columna difficulty (no útil)
        full_df = full_df.drop(['difficulty'], axis=1)
        
        # Identificar columnas categóricas
        categorical_cols = ['protocol_type', 'service', 'flag']
        
        # Encoding de categóricas
        for col in categorical_cols:
            le = LabelEncoder()
            full_df[col] = le.fit_transform(full_df[col])
            self.label_encoders[col] = le
        
        # Separar features y labels
        X = full_df.drop(['label'], axis=1).values
        y = full_df['label'].values
        
        # Normalización
        X = self.scaler.fit_transform(X)
        
        # Dividir train/test según tamaño original
        train_size = len(train_df)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Reshape para CNN (añadir dimensión de canal)
        X_train = np.expand_dims(X_train, axis=2)
        X_test = np.expand_dims(X_test, axis=2)
        
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        print(f"Train labels - Normal: {np.sum(y_train == 0)}, Attack: {np.sum(y_train == 1)}")
        print(f"Test labels - Normal: {np.sum(y_test == 0)}, Attack: {np.sum(y_test == 1)}")
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape):
        """
        Arquitectura CNN profunda optimizada para IDS
        - 4 bloques convolucionales con filtros progresivos
        - BatchNormalization después de cada conv
        - Dropout adaptativo (0.2 → 0.3 → 0.4)
        - Regularización L2
        - Global Average Pooling para reducir parámetros
        """
        model = keras.Sequential([
            # Bloque 1: Detección de patrones básicos
            layers.Conv1D(32, 3, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.001),
                         input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv1D(32, 3, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),
            
            # Bloque 2: Patrones intermedios
            layers.Conv1D(64, 3, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Conv1D(64, 3, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            # Bloque 3: Patrones complejos
            layers.Conv1D(128, 3, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Conv1D(128, 3, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.4),
            
            # Bloque 4: Features de alto nivel
            layers.Conv1D(256, 3, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),  # Más robusto que Flatten
            
            # Capas densas
            layers.Dense(128, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(64, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Capa de salida
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compilar con Focal Loss y métricas orientadas a recall
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=FocalLoss(alpha=0.4, gamma=2.0),  # Focal loss balanceado
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=128):
        """
        Entrena el modelo con callbacks optimizados
        """
        # Class weights balanceados para recall sin sacrificar precision
        # Normal: 0.67, Attack: 1.33 (ratio 1:2)
        class_weight = {0: 0.67, 1: 1.33}
        
        callbacks = [
            # Early stopping con paciencia mayor para Focal Loss
            EarlyStopping(
                monitor='val_recall',
                patience=10,
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            # Reducir LR cuando recall se estanque
            ReduceLROnPlateau(
                monitor='val_recall',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                mode='max',
                verbose=1
            ),
            # Guardar mejor modelo según recall
            ModelCheckpoint(
                self.model_path if self.model_path else 'best_cnn_v2_model.h5',
                monitor='val_recall',
                mode='max',
                save_best_only=True,
                verbose=1
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test, threshold=0.35):
        """
        Evalúa el modelo con threshold optimizado para recall
        """
        # Predicciones de probabilidad
        y_pred_proba = self.model.predict(X_test)
        
        # Aplicar threshold
        y_pred = (y_pred_proba > threshold).astype(int)
        
        from sklearn.metrics import classification_report, confusion_matrix
        
        print(f"\n{'='*60}")
        print(f"Evaluación con threshold={threshold}")
        print(f"{'='*60}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print(f"\nTN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        # Calcular recall de ataques
        attack_recall = cm[1,1] / (cm[1,1] + cm[1,0])
        print(f"\n🎯 RECALL DE ATAQUES: {attack_recall*100:.2f}%")
        print(f"   Ataques detectados: {cm[1,1]} de {cm[1,1] + cm[1,0]}")
        print(f"   Ataques perdidos: {cm[1,0]}")
        
        return y_pred, y_pred_proba
    
    def save_scaler(self, path='scaler_cnn_v2.pkl'):
        """Guarda el scaler"""
        joblib.dump(self.scaler, path)
        print(f"Scaler guardado en: {path}")
    
    def load_model(self, model_path, scaler_path='scaler_cnn_v2.pkl'):
        """Carga modelo y scaler"""
        self.model = keras.models.load_model(
            model_path,
            custom_objects={'FocalLoss': FocalLoss}
        )
        self.scaler = joblib.load(scaler_path)
        print(f"Modelo y scaler cargados correctamente")


if __name__ == "__main__":
    # Paths del dataset
    TRAIN_PATH = "C:/Users/Diego/Downloads/NSL_KDD-master/NSL_KDD-master/KDDTrain+.csv"
    TEST_PATH = "C:/Users/Diego/Downloads/NSL_KDD-master/NSL_KDD-master/KDDTest+.csv"
    
    # Crear modelo
    ids_model = IDSModelCNN_v2(model_path='best_cnn_v2_model.h5')
    
    # Cargar y preprocesar
    print("Cargando y preprocesando datos...")
    X_train, X_test, y_train, y_test = ids_model.load_and_preprocess(TRAIN_PATH, TEST_PATH)
    
    # Dividir train en train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Construir modelo
    print("\nConstruyendo modelo CNN v2...")
    ids_model.build_model(input_shape=(X_train.shape[1], 1))
    ids_model.model.summary()
    
    # Entrenar
    print("\nIniciando entrenamiento...")
    history = ids_model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=128)
    
    # Guardar scaler
    ids_model.save_scaler('scaler_cnn_v2.pkl')
    
    # Evaluar con diferentes thresholds
    print("\n" + "="*60)
    print("EVALUACIÓN FINAL")
    print("="*60)
    
    for threshold in [0.7, 0.65, 0.6, 0.55, 0.5]:
        ids_model.evaluate(X_test, y_test, threshold=threshold)
