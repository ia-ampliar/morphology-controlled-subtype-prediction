import os
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import Model

import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_confusion_matrix(cm, class_names: List[str], figsize: Tuple = (10, 7), 
                           fontsize: int = 11, output_path: str = None,
                           architecture_name: str = None):
    """
    Cria e salva matriz de confusão visualizada.
    
    Args:
        cm (np.ndarray): Matriz de confusão
        class_names (List[str]): Nomes das classes
        figsize (Tuple): Tamanho da figura
        fontsize (int): Tamanho da fonte
        output_path (str): Caminho para salvar (default: with timestamp)
        architecture_name (str): Nome da arquitetura (para nome do arquivo)
    """
    if output_path is None:
        output_path = f'outputs/{architecture_name}_confusion_matrix_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png'
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    fig = plt.figure(figsize=figsize)
    
    try:
        heatmap = sns.heatmap(df_cm, cmap="YlGnBu", annot=True, fmt="d")
    except ValueError:
        raise ValueError("[ERROR] - Confusion matrix values must be integers.")
    
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=30, ha='right', fontsize=fontsize)
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Matriz de confusão salva em: {output_path}")


def save_roc_curve(y_true, y_scores, class_names: List[str] = None,
                   figsize: Tuple = (8, 8), output_path: str = None,
                   architecture_name: str = None):
    """
    Cria e salva o gráfico ROC/AUC.

    Args:
        y_true: Labels verdadeiros (inteiros ou one-hot)
        y_scores: Probabilidades de saída do modelo
        class_names (List[str], optional): Nomes das classes para legenda
        figsize (Tuple, optional): Tamanho da figura
        output_path (str, optional): Caminho de saída do arquivo
        architecture_name (str, optional): Nome da arquitetura usado no filename
    """
    if y_scores.ndim == 1 or (y_scores.ndim == 2 and y_scores.shape[1] == 1):
        y_scores = np.hstack([1 - y_scores.reshape(-1, 1), y_scores.reshape(-1, 1)])

    n_classes = y_scores.shape[1]

    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true_bin = y_true
    else:
        if n_classes == 2:
            y_true_bin = label_binarize(y_true, classes=[0, 1])
            if y_true_bin.shape[1] == 1:
                y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        else:
            y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    if y_true_bin.shape[1] != n_classes:
        raise ValueError(
            f"Class mismatch: y_true has {y_true_bin.shape[1]} binarized columns but y_scores has {n_classes} classes."
        )

    if output_path is None:
        output_path = f'outputs/{architecture_name}_roc_curve_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png'

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    plt.figure(figsize=figsize)

    all_fpr = np.unique(np.concatenate([
        roc_curve(y_true_bin[:, i], y_scores[:, i])[0]
        for i in range(n_classes)
    ]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
        roc_auc = auc(fpr, tpr)
        label = f"{class_names[i]} (AUC={roc_auc:.3f})" if class_names else f"Class {i} (AUC={roc_auc:.3f})"
        plt.plot(fpr, tpr, lw=1.5, label=label)

    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr, color='black', linestyle='--', linewidth=2,
             label=f'Macro-average (AUC={macro_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', alpha=0.6)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    logger.info(f"ROC curve salva em: {output_path}")


def evaluate_model(model: Model, test_data, class_names: List[str], architecture_name: str = None) -> Dict:
    """
    Avalia modelo no dataset de teste e gera métricas.
    
    Args:
        model (Model): Modelo treinado
        test_data: Dataset de teste
        class_names (List[str]): Nomes das classes
        architecture_name (str): Nome da arquitetura

    Returns:
        Dict: Dicionário com métricas (loss, acc, f1, precision, recall, auc)
    """
    logger.info("=== Avaliação no Conjunto de Teste ===")
    
    num_test = test_data.samples
    
    # Predições
    Y_pred = model.predict(test_data, steps=num_test, verbose=1)
    test_preds = np.argmax(Y_pred, axis=-1)
    test_trues = test_data.classes
    
    # Matriz de confusão
    cm = confusion_matrix(test_trues, test_preds)
    print_confusion_matrix(cm, class_names, figsize=(5, 5), fontsize=11, architecture_name=architecture_name)
    
    # ROC curve
    save_roc_curve(test_trues, Y_pred, class_names, architecture_name=architecture_name)
    
    # Métricas
    loss, acc, f1_score, prec, rec, aroc = model.evaluate(test_data, verbose=1)
    
    metrics = {
        'loss': loss,
        'accuracy': acc,
        'f1_score': f1_score,
        'precision': prec,
        'recall': rec,
        'auc': aroc
    }
    
    # Log
    logger.info(f"[METRIC] - Test accuracy: {acc:.4f}")
    logger.info(f"[METRIC] - Test F1-Score: {f1_score:.4f}")
    logger.info(f"[METRIC] - Test Precision: {prec:.4f}")
    logger.info(f"[METRIC] - Test Recall: {rec:.4f}")
    logger.info(f"[METRIC] - Test AUC: {aroc:.4f}")
    logger.info(f"[METRIC] - Test Loss: {loss:.4f}")
    
    return metrics