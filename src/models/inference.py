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
        output_path = f'outputs/metrics/{architecture_name}_confusion_matrix_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png'
    
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
    
    # changes true and predicted labels from 0 and 1 to class names
    test_trues_names = [class_names[i] for i in test_trues]
    test_preds_names = [class_names[i] for i in test_preds]

    # cria um data frame com o endereço das imagens, as classes verdadeiras e preditas e probabilidades preditas
    test_filenames = test_data.filenames
    test_df = pd.DataFrame({
        'filename': test_filenames,
        'true_label': test_trues_names,
        'predicted_label': test_preds_names,
        **{class_names[i]: Y_pred[:, i] for i in range(len(class_names))}
    })
    
    # save test_df to csv
    output_csv_path = f'outputs/dataframes/{architecture_name}_test_predictions_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
    os.makedirs(os.path.dirname(output_csv_path) or '.', exist_ok=True)
    test_df.to_csv(output_csv_path, index=False)
    
    # Log
    logger.info(f"[METRIC] - Test accuracy: {acc:.4f}")
    logger.info(f"[METRIC] - Test F1-Score: {f1_score:.4f}")
    logger.info(f"[METRIC] - Test Precision: {prec:.4f}")
    logger.info(f"[METRIC] - Test Recall: {rec:.4f}")
    logger.info(f"[METRIC] - Test AUC: {aroc:.4f}")
    logger.info(f"[METRIC] - Test Loss: {loss:.4f}")
    
    return metrics