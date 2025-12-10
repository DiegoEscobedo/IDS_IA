"""
Script de Comparación Completa - Baseline vs Mejoras
Genera reportes detallados y visualizaciones para presentación
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Importar modelos v2
from IDSModelCNN_v2 import IDSModelCNN_v2, FocalLoss
from IDSModelLSTM_v2 import IDSModelLSTM_v2, AttentionLayer
from ensemble_v2 import EnsembleIDS_v2


def evaluate_model(model, X_test, y_test, threshold=0.35, model_name="Model"):
    """
    Evalúa un modelo y retorna métricas
    """
    # Predicciones
    if hasattr(model, 'predict_ensemble'):
        # Es ensemble
        y_pred, y_pred_proba, _, _ = model.predict_ensemble(X_test, threshold=threshold, method='stacking')
    else:
        # Modelo individual
        y_pred_proba = model.model.predict(np.expand_dims(X_test, axis=2), verbose=0).flatten()
        y_pred = (y_pred_proba > threshold).astype(int)
    
    # Calcular métricas
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calcular AUC-ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Calcular AUC-PR
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall_curve, precision_curve)
    
    return {
        'model_name': model_name,
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'attacks_detected': int(tp),
        'attacks_total': int(tp + fn),
        'attacks_missed': int(fn),
        'y_pred_proba': y_pred_proba,
        'fpr': fpr,
        'tpr': tpr
    }


def create_comparison_table(results):
    """
    Crea tabla comparativa de resultados
    """
    print("\n" + "="*100)
    print("TABLA COMPARATIVA - BASELINE vs MEJORAS")
    print("="*100)
    
    # Crear DataFrame
    df = pd.DataFrame([
        {
            'Modelo': r['model_name'],
            'Threshold': f"{r['threshold']:.2f}",
            'Accuracy': f"{r['accuracy']*100:.2f}%",
            'Precision': f"{r['precision']*100:.2f}%",
            'Recall ⭐': f"{r['recall']*100:.2f}%",
            'F1-Score': f"{r['f1']*100:.2f}%",
            'ROC-AUC': f"{r['roc_auc']:.4f}",
            'Ataques Detectados': f"{r['attacks_detected']}/{r['attacks_total']}",
            'Ataques Perdidos': r['attacks_missed']
        }
        for r in results
    ])
    
    print(df.to_string(index=False))
    
    # Guardar como CSV
    df.to_csv('comparacion_modelos.csv', index=False)
    print("\n✅ Tabla guardada: comparacion_modelos.csv")
    
    return df


def plot_comparison_metrics(results, save_path='comparacion_metricas.png'):
    """
    Gráfico de barras comparando métricas clave
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    models = [r['model_name'] for r in results]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    
    # Recall (más importante)
    axes[0, 0].bar(models, [r['recall']*100 for r in results], color=colors)
    axes[0, 0].axhline(85, color='green', linestyle='--', linewidth=2, label='Target (85%)')
    axes[0, 0].axhline(63, color='red', linestyle='--', linewidth=2, label='Baseline (63%)')
    axes[0, 0].set_ylabel('Recall (%)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('🎯 RECALL (Métrica Crítica para IDS)', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Precision
    axes[0, 1].bar(models, [r['precision']*100 for r in results], color=colors)
    axes[0, 1].set_ylabel('Precision (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Precision (Reducir Falsos Positivos)', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # F1-Score
    axes[1, 0].bar(models, [r['f1']*100 for r in results], color=colors)
    axes[1, 0].set_ylabel('F1-Score (%)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('F1-Score (Balance Precision-Recall)', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Ataques Perdidos
    axes[1, 1].bar(models, [r['attacks_missed'] for r in results], color=colors)
    axes[1, 1].set_ylabel('Ataques Perdidos', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('❌ Ataques NO Detectados (Minimizar)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico guardado: {save_path}")
    plt.close()


def plot_roc_comparison(results, save_path='comparacion_roc.png'):
    """
    Curvas ROC comparativas
    """
    plt.figure(figsize=(12, 10))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    
    for i, r in enumerate(results):
        plt.plot(r['fpr'], r['tpr'], 
                label=f"{r['model_name']} (AUC={r['roc_auc']:.4f})",
                linewidth=2.5 if 'Ensemble' in r['model_name'] else 2,
                linestyle='--' if 'Ensemble' in r['model_name'] else '-',
                color=colors[i % len(colors)])
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5000)', linewidth=1)
    
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate (Recall)', fontsize=14, fontweight='bold')
    plt.title('Comparación Curvas ROC - Todos los Modelos', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Curva ROC guardada: {save_path}")
    plt.close()


def plot_confusion_matrices(results, save_path='confusion_matrices.png'):
    """
    Matrices de confusión comparativas
    """
    n_models = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, r in enumerate(results):
        cm = np.array([[r['tn'], r['fp']], 
                       [r['fn'], r['tp']]])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'],
                   ax=axes[i], cbar=False,
                   annot_kws={'fontsize': 14, 'fontweight': 'bold'})
        
        axes[i].set_title(f"{r['model_name']}\nRecall: {r['recall']*100:.1f}%", 
                         fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Real', fontsize=11)
        axes[i].set_xlabel('Predicción', fontsize=11)
    
    # Ocultar ejes sobrantes
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Matrices de confusión guardadas: {save_path}")
    plt.close()


def generate_improvement_report(results):
    """
    Genera reporte de mejoras comparando modelos
    """
    print("\n" + "="*100)
    print("REPORTE DE COMPARACIÓN - CNN vs LSTM vs ENSEMBLE")
    print("="*100)
    
    # Comparar todos los modelos
    for i, r in enumerate(results):
        print(f"\n🔹 {r['model_name']}:")
        print(f"   Recall: {r['recall']*100:.2f}%")
        print(f"   Precision: {r['precision']*100:.2f}%")
        print(f"   F1-Score: {r['f1']*100:.2f}%")
        print(f"   Accuracy: {r['accuracy']*100:.2f}%")
        print(f"   Ataques detectados: {r['attacks_detected']}/{r['attacks_total']}")
        print(f"   Ataques perdidos: {r['attacks_missed']}")
    
    # Identificar mejor modelo
    best_model = max(results, key=lambda x: x['f1'])  # Usar F1 como métrica principal
    
    print("\n" + "="*100)
    print(f"🏆 MEJOR MODELO: {best_model['model_name']}")
    print("="*100)
    print(f"   Recall: {best_model['recall']*100:.2f}%")
    print(f"   Precision: {best_model['precision']*100:.2f}%")
    print(f"   F1-Score: {best_model['f1']*100:.2f}%")
    print(f"   Accuracy: {best_model['accuracy']*100:.2f}%")
    print(f"   Ataques detectados: {best_model['attacks_detected']}/{best_model['attacks_total']}")
    
    return best_model


def save_results_json(results, filename='resultados_completos.json'):
    """
    Guarda resultados en JSON para análisis posterior
    """
    # Convertir arrays numpy a listas
    results_clean = []
    for r in results:
        r_copy = r.copy()
        r_copy['fpr'] = r['fpr'].tolist()
        r_copy['tpr'] = r['tpr'].tolist()
        r_copy['y_pred_proba'] = r['y_pred_proba'].tolist()
        results_clean.append(r_copy)
    
    with open(filename, 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    print(f"\n✅ Resultados guardados en JSON: {filename}")


def main():
    """
    Ejecuta comparación completa
    """
    print("="*100)
    print("ANÁLISIS COMPARATIVO COMPLETO - BASELINE vs MEJORAS v2")
    print("="*100)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Cargar datos
    from train_with_smote import load_and_preprocess_data
    from sklearn.preprocessing import StandardScaler
    
    TRAIN_PATH = "C:/Users/Diego/Downloads/NSL_KDD-master/NSL_KDD-master/KDDTrain+.csv"
    TEST_PATH = "C:/Users/Diego/Downloads/NSL_KDD-master/NSL_KDD-master/KDDTest+.csv"
    
    print("\n[1/3] Cargando datos de prueba...")
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(TRAIN_PATH, TEST_PATH)
    
    results = []
    
    # Evaluar CNN v2 con SMOTE
    print("\n[2/3] Evaluando CNN v2 + SMOTE...")
    try:
        cnn_v2 = IDSModelCNN_v2()
        cnn_v2.load_model('best_cnn_v2_smote.h5', 'scaler_best_cnn_v2_smote.pkl')
        results.append(evaluate_model(cnn_v2, X_test, y_test, 
                                     threshold=0.6, model_name="CNN v2 + SMOTE"))
    except Exception as e:
        print(f"⚠️  CNN v2 no encontrado: {e}")
    
    # Evaluar LSTM v2 con SMOTE
    print("\n[3/3] Evaluando LSTM v2 + SMOTE...")
    try:
        lstm_v2 = IDSModelLSTM_v2()
        lstm_v2.load_model('best_lstm_v2_smote.h5', 'scaler_best_lstm_v2_smote.pkl')
        results.append(evaluate_model(lstm_v2, X_test, y_test, 
                                     threshold=0.6, model_name="LSTM v2 + SMOTE"))
    except Exception as e:
        print(f"⚠️  LSTM v2 no encontrado: {e}")
    
    # Evaluar Ensemble v2 con threshold óptimo
    print("\n[4/4] Evaluando Ensemble v2...")
    try:
        ensemble = EnsembleIDS_v2(
            cnn_model_path='best_cnn_v2_smote.h5',
            lstm_model_path='best_lstm_v2_smote.h5',
            cnn_scaler_path='scaler_best_cnn_v2_smote.pkl',
            lstm_scaler_path='scaler_best_lstm_v2_smote.pkl'
        )
        ensemble.load_models()
        ensemble.train_meta_learner(X_train, y_train)
        
        results.append(evaluate_model(ensemble, X_test, y_test, 
                                     threshold=0.4, model_name="Ensemble v2 (Stacking)"))
    except Exception as e:
        print(f"⚠️  Ensemble v2 no encontrado: {e}")
    
    if not results:
        print("\n❌ No se encontraron modelos entrenados. Ejecuta primero train_with_smote.py")
        return
    
    # Generar visualizaciones y reportes
    print("\n" + "="*100)
    print("GENERANDO REPORTES Y VISUALIZACIONES")
    print("="*100)
    
    create_comparison_table(results)
    plot_comparison_metrics(results)
    plot_roc_comparison(results)
    plot_confusion_matrices(results)
    generate_improvement_report(results)
    save_results_json(results)
    
    print("\n" + "="*100)
    print("✅ ANÁLISIS COMPLETO FINALIZADO")
    print("="*100)
    print("\nArchivos generados:")
    print("  📊 comparacion_modelos.csv")
    print("  📈 comparacion_metricas.png")
    print("  📉 comparacion_roc.png")
    print("  🔲 confusion_matrices.png")
    print("  💾 resultados_completos.json")
    print("\n¡Listos para presentación! 🎉")


if __name__ == "__main__":
    main()
