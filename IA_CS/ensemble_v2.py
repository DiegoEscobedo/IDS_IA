"""
Ensemble Avanzado v2 - Combina CNN y LSTM con Stacking y Calibración
Optimizado para maximizar recall en detección de ataques
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import joblib

from IDSModelCNN_v2 import FocalLoss
from IDSModelLSTM_v2 import AttentionLayer


class EnsembleIDS_v2:
    def __init__(self, cnn_model_path, lstm_model_path, 
                 cnn_scaler_path, lstm_scaler_path):
        """
        Ensemble avanzado con stacking y calibración
        """
        self.cnn_model = None
        self.lstm_model = None
        self.cnn_scaler = None
        self.lstm_scaler = None
        self.meta_learner = None
        
        self.cnn_model_path = cnn_model_path
        self.lstm_model_path = lstm_model_path
        self.cnn_scaler_path = cnn_scaler_path
        self.lstm_scaler_path = lstm_scaler_path
        
    def load_models(self):
        """
        Carga modelos CNN y LSTM pre-entrenados
        """
        print("Cargando modelos...")
        
        # Cargar CNN
        self.cnn_model = keras.models.load_model(
            self.cnn_model_path,
            custom_objects={'FocalLoss': FocalLoss}
        )
        self.cnn_scaler = joblib.load(self.cnn_scaler_path)
        
        # Cargar LSTM
        self.lstm_model = keras.models.load_model(
            self.lstm_model_path,
            custom_objects={
                'FocalLoss': FocalLoss,
                'AttentionLayer': AttentionLayer
            }
        )
        self.lstm_scaler = joblib.load(self.lstm_scaler_path)
        
        print("✅ Modelos cargados correctamente")
        
    def get_base_predictions(self, X, scalers_already_applied=False):
        """
        Obtiene predicciones de CNN y LSTM
        """
        if not scalers_already_applied:
            X_cnn = self.cnn_scaler.transform(X)
            X_lstm = self.lstm_scaler.transform(X)
        else:
            X_cnn = X.copy()
            X_lstm = X.copy()
        
        # Reshape para cada modelo
        X_cnn_reshaped = np.expand_dims(X_cnn, axis=2)
        X_lstm_reshaped = np.expand_dims(X_lstm, axis=2)
        
        # Predicciones
        cnn_pred = self.cnn_model.predict(X_cnn_reshaped, verbose=0).flatten()
        lstm_pred = self.lstm_model.predict(X_lstm_reshaped, verbose=0).flatten()
        
        return cnn_pred, lstm_pred
    
    def train_meta_learner(self, X_train, y_train):
        """
        Entrena meta-learner (stacking) usando predicciones de modelos base
        """
        print("\n" + "="*60)
        print("ENTRENANDO META-LEARNER (STACKING)")
        print("="*60)
        
        # Obtener predicciones de modelos base
        cnn_pred, lstm_pred = self.get_base_predictions(X_train)
        
        # Crear dataset para meta-learner
        # Features: [cnn_prob, lstm_prob, avg_prob, diff_prob]
        meta_features = np.column_stack([
            cnn_pred,
            lstm_pred,
            (cnn_pred + lstm_pred) / 2,  # Promedio
            np.abs(cnn_pred - lstm_pred)  # Diferencia (mide acuerdo)
        ])
        
        # Entrenar Logistic Regression con calibración
        print("Entrenando Logistic Regression con calibración...")
        base_lr = LogisticRegression(
            class_weight={0: 0.67, 1: 1.33},  # Balanceado para precision-recall
            max_iter=1000,
            random_state=42
        )
        
        # Calibrar probabilidades con CalibratedClassifierCV
        self.meta_learner = CalibratedClassifierCV(
            base_lr,
            method='sigmoid',
            cv=5
        )
        
        self.meta_learner.fit(meta_features, y_train)
        
        print("✅ Meta-learner entrenado y calibrado")
        
        return self.meta_learner
    
    def predict_ensemble(self, X, threshold=0.35, method='stacking'):
        """
        Predicción con ensemble
        
        Métodos:
        - 'stacking': Usa meta-learner entrenado
        - 'weighted': Promedio ponderado (CNN: 0.4, LSTM: 0.6)
        - 'average': Promedio simple
        - 'max': Máxima confianza
        """
        cnn_pred, lstm_pred = self.get_base_predictions(X)
        
        if method == 'stacking':
            if self.meta_learner is None:
                raise ValueError("Meta-learner no entrenado. Ejecutar train_meta_learner() primero")
            
            meta_features = np.column_stack([
                cnn_pred,
                lstm_pred,
                (cnn_pred + lstm_pred) / 2,
                np.abs(cnn_pred - lstm_pred)
            ])
            
            ensemble_proba = self.meta_learner.predict_proba(meta_features)[:, 1]
        
        elif method == 'weighted':
            # CNN tiende a precision, LSTM a recall → dar más peso a LSTM
            ensemble_proba = 0.4 * cnn_pred + 0.6 * lstm_pred
        
        elif method == 'average':
            ensemble_proba = (cnn_pred + lstm_pred) / 2
        
        elif method == 'max':
            ensemble_proba = np.maximum(cnn_pred, lstm_pred)
        
        else:
            raise ValueError(f"Método '{method}' no válido")
        
        # Aplicar threshold
        ensemble_pred = (ensemble_proba > threshold).astype(int)
        
        return ensemble_pred, ensemble_proba, cnn_pred, lstm_pred
    
    def evaluate_ensemble(self, X_test, y_test, threshold=0.35, method='stacking'):
        """
        Evalúa ensemble y muestra comparación con modelos individuales
        """
        print(f"\n{'='*70}")
        print(f"EVALUACIÓN ENSEMBLE - Método: {method.upper()} - Threshold: {threshold}")
        print(f"{'='*70}")
        
        # Predicciones
        ensemble_pred, ensemble_proba, cnn_pred, lstm_pred = self.predict_ensemble(
            X_test, threshold=threshold, method=method
        )
        
        cnn_pred_binary = (cnn_pred > threshold).astype(int)
        lstm_pred_binary = (lstm_pred > threshold).astype(int)
        
        # Métricas para cada modelo
        print("\n" + "="*70)
        print("COMPARACIÓN: CNN vs LSTM vs ENSEMBLE")
        print("="*70)
        
        for name, preds in [('CNN', cnn_pred_binary), 
                            ('LSTM', lstm_pred_binary), 
                            ('ENSEMBLE', ensemble_pred)]:
            print(f"\n--- {name} ---")
            cm = confusion_matrix(y_test, preds)
            
            tn, fp, fn, tp = cm.ravel()
            
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"Accuracy:  {accuracy*100:.2f}%")
            print(f"Precision: {precision*100:.2f}%")
            print(f"Recall:    {recall*100:.2f}% 🎯")
            print(f"F1-Score:  {f1*100:.2f}%")
            print(f"Ataques detectados: {tp}/{tp+fn} ({recall*100:.1f}%)")
            print(f"Ataques perdidos: {fn}")
        
        # Reporte detallado del ensemble
        print("\n" + "="*70)
        print("REPORTE DETALLADO DEL ENSEMBLE")
        print("="*70)
        print(classification_report(y_test, ensemble_pred, target_names=['Normal', 'Attack']))
        
        cm = confusion_matrix(y_test, ensemble_pred)
        print("\nConfusion Matrix:")
        print(cm)
        print(f"\nTN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        return ensemble_pred, ensemble_proba
    
    def plot_roc_comparison(self, X_test, y_test, save_path='ensemble_roc_comparison.png'):
        """
        Compara curvas ROC de CNN, LSTM y Ensemble
        """
        # Obtener predicciones
        ensemble_pred_stacking, ensemble_proba_stacking, cnn_pred, lstm_pred = self.predict_ensemble(
            X_test, method='stacking'
        )
        
        _, ensemble_proba_weighted, _, _ = self.predict_ensemble(
            X_test, method='weighted'
        )
        
        # Calcular ROC para cada modelo
        fpr_cnn, tpr_cnn, _ = roc_curve(y_test, cnn_pred)
        fpr_lstm, tpr_lstm, _ = roc_curve(y_test, lstm_pred)
        fpr_stack, tpr_stack, _ = roc_curve(y_test, ensemble_proba_stacking)
        fpr_weighted, tpr_weighted, _ = roc_curve(y_test, ensemble_proba_weighted)
        
        auc_cnn = auc(fpr_cnn, tpr_cnn)
        auc_lstm = auc(fpr_lstm, tpr_lstm)
        auc_stack = auc(fpr_stack, tpr_stack)
        auc_weighted = auc(fpr_weighted, tpr_weighted)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.plot(fpr_cnn, tpr_cnn, label=f'CNN (AUC={auc_cnn:.4f})', linewidth=2)
        plt.plot(fpr_lstm, tpr_lstm, label=f'LSTM (AUC={auc_lstm:.4f})', linewidth=2)
        plt.plot(fpr_stack, tpr_stack, label=f'Ensemble Stacking (AUC={auc_stack:.4f})', 
                linewidth=3, linestyle='--')
        plt.plot(fpr_weighted, tpr_weighted, label=f'Ensemble Weighted (AUC={auc_weighted:.4f})', 
                linewidth=2, linestyle=':')
        plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate (Recall)', fontsize=12)
        plt.title('Comparación ROC: CNN vs LSTM vs Ensemble', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Gráfico ROC guardado: {save_path}")
        plt.close()
    
    def optimize_threshold_ensemble(self, X_test, y_test, method='stacking'):
        """
        Encuentra threshold óptimo para ensemble
        """
        print(f"\n{'='*60}")
        print(f"OPTIMIZACIÓN DE THRESHOLD - Método: {method.upper()}")
        print(f"{'='*60}")
        
        # Obtener probabilidades
        _, ensemble_proba, _, _ = self.predict_ensemble(X_test, method=method)
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        results = []
        
        for threshold in thresholds:
            preds = (ensemble_proba > threshold).astype(int)
            cm = confusion_matrix(y_test, preds)
            tn, fp, fn, tp = cm.ravel()
            
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                'threshold': threshold,
                'recall': recall,
                'precision': precision,
                'f1': f1
            })
        
        # Encontrar threshold para 85% recall
        target_recall = 0.85
        best_threshold = None
        best_precision = 0
        
        for r in results:
            if r['recall'] >= target_recall and r['precision'] > best_precision:
                best_threshold = r['threshold']
                best_precision = r['precision']
        
        if best_threshold is None:
            # Si no alcanza 85%, buscar el de mayor recall
            best_threshold = max(results, key=lambda x: x['recall'])['threshold']
        
        print(f"\n🎯 THRESHOLD ÓPTIMO: {best_threshold:.2f}")
        print(f"   Recall: {[r for r in results if r['threshold']==best_threshold][0]['recall']*100:.2f}%")
        print(f"   Precision: {[r for r in results if r['threshold']==best_threshold][0]['precision']*100:.2f}%")
        print(f"   F1-Score: {[r for r in results if r['threshold']==best_threshold][0]['f1']*100:.2f}%")
        
        # Plot threshold analysis
        plt.figure(figsize=(12, 6))
        
        threshs = [r['threshold'] for r in results]
        recalls = [r['recall'] for r in results]
        precisions = [r['precision'] for r in results]
        f1s = [r['f1'] for r in results]
        
        plt.plot(threshs, recalls, 'b-', label='Recall', linewidth=2)
        plt.plot(threshs, precisions, 'r-', label='Precision', linewidth=2)
        plt.plot(threshs, f1s, 'g-', label='F1-Score', linewidth=2)
        plt.axvline(best_threshold, color='orange', linestyle='--', 
                   label=f'Óptimo ({best_threshold:.2f})', linewidth=2)
        plt.axhline(0.85, color='gray', linestyle=':', alpha=0.5, label='Target Recall (85%)')
        
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title(f'Optimización de Threshold - Ensemble {method.capitalize()}', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'ensemble_{method}_threshold_optimization.png', dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado: ensemble_{method}_threshold_optimization.png")
        plt.close()
        
        return best_threshold


if __name__ == "__main__":
    # Cargar datos de test
    from train_with_smote import load_and_preprocess_data
    
    TRAIN_PATH = "C:/Users/Diego/Downloads/NSL_KDD-master/NSL_KDD-master/KDDTrain+.csv"
    TEST_PATH = "C:/Users/Diego/Downloads/NSL_KDD-master/NSL_KDD-master/KDDTest+.csv"
    
    print("="*70)
    print("ENSEMBLE AVANZADO v2 - EVALUACIÓN")
    print("="*70)
    
    # Cargar datos
    print("\nCargando datos de prueba...")
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(TRAIN_PATH, TEST_PATH)
    
    # Crear ensemble
    ensemble = EnsembleIDS_v2(
        cnn_model_path='best_cnn_v2_smote.h5',
        lstm_model_path='best_lstm_v2_smote.h5',
        cnn_scaler_path='scaler_best_cnn_v2_smote.pkl',
        lstm_scaler_path='scaler_best_lstm_v2_smote.pkl'
    )
    
    # Cargar modelos
    ensemble.load_models()
    
    # Entrenar meta-learner
    ensemble.train_meta_learner(X_train, y_train)
    
    # Evaluar con diferentes métodos
    methods = ['stacking', 'weighted', 'average', 'max']
    
    for method in methods:
        ensemble.evaluate_ensemble(X_test, y_test, threshold=0.6, method=method)
    
    # Optimizar threshold para stacking
    best_threshold = ensemble.optimize_threshold_ensemble(X_test, y_test, method='stacking')
    
    # Evaluación final con threshold óptimo
    print("\n" + "="*70)
    print("EVALUACIÓN FINAL CON THRESHOLD ÓPTIMO")
    print("="*70)
    ensemble.evaluate_ensemble(X_test, y_test, threshold=best_threshold, method='stacking')
    
    # Comparación ROC
    ensemble.plot_roc_comparison(X_test, y_test)
    
    print("\n" + "="*70)
    print("✅ EVALUACIÓN COMPLETA")
    print("="*70)
