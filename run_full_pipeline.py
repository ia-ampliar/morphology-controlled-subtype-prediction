# -*- coding: utf-8 -*-
"""
MobileNetV2 - Gastric Cancer Molecular Subtype Prediction

Módulo para treinar e avaliar MobileNetV2 em histopatologia de câncer gástrico
com controle de morfologia. Implementa Fine Tuning em duas fases com early stopping.

Requirimentos:
    - python=3.10.12
    - tensorflow=2.15.0
    - numpy=1.25.2
    - pandas=2.0.3
    - seaborn=0.12.2
    - matplotlib=3.7.1
    - opencv-python=4.8.0.76
    - scikit-learn=1.2.2
"""

import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import platform
from datetime import datetime
from typing import Tuple, List, Dict

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, NASNetMobile, ResNet50V2, VGG19, InceptionV3, DenseNet201
from sklearn.metrics import confusion_matrix

import logging

from src.models.dataset import setup_data_directories, load_all_datasets
from src.utils.utils import get_class_info, save_metrics_to_file, print_environment_info
from src.models.build_model import build_model
from src.models.train_model import train_phase1, train_phase2
from src.models.inference import evaluate_model

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURAÇÃO E INICIALIZAÇÃO
# ============================================================================


def load_config() -> Dict:
    """
    Carrega configurações do modelo e treinamento.
    
    Returns:
        dict: Dicionário com configurações
            - base_path: Caminho base para os datasets
            - img_shape: Formato das imagens (224, 224, 3)
            - batch_size: Tamanho do batch
            - epochs: Número máximo de épocas
            - base_learning_rate: Taxa de aprendizado inicial
    """
    config = {
        'base_path': 'sanity_test_dataset/',
        'img_shape': (224, 224, 3),
        'batch_size': 32,
        'epochs': 5,
        'base_learning_rate': 0.0001,
        'patience': 7,
        'min_delta': 0.001,
        'phase1_ratio': 0.4,  # 40% das épocas na fase 1
        'unfreeze_ratio': 0.3,  # Descongelar 30% das camadas na fase 2
    }
    return config



# ============================================================================
# PIPELINE COMPLETO
# ============================================================================

def run_full_pipeline(architecture_name: str = "MobileNetV2", base_path: str = "sanity_test_dataset/") -> Tuple[Model, Dict]:
    """
    Executa pipeline completo: carregamento, construção, treinamento e avaliação.
    """
    logger.info("="*80)
    logger.info(f"INICIANDO PIPELINE DE TREINAMENTO - {architecture_name}")
    logger.info("="*80)
    
    # Setup
    print_environment_info()
    config = load_config()
    
    # Dados
    train_dir, val_dir, test_dir = setup_data_directories(config, base_path)
    train_data, val_data, test_data = load_all_datasets(
        train_dir, val_dir, test_dir, config['batch_size']
    )
    class_names, num_classes = get_class_info(val_data)
    
    # Modelo
    model, basemodel = build_model(config['img_shape'], num_classes, architecture_name)
    
    # Treinamento Fase 1
    history_phase1 = train_phase1(
        model, 
        basemodel, 
        train_data, 
        val_data, 
        config, 
        architecture_name
    )
    
    # Treinamento Fase 2
    history_phase2 = train_phase2(
        architecture_name=architecture_name, 
        model=model, 
        basemodel=basemodel, 
        train_data=train_data, 
        val_data=val_data, 
        config=config, 
        history_phase1=history_phase1
    )
    
    # Avaliação
    metrics = evaluate_model(model, test_data, class_names, architecture_name=architecture_name)
    save_metrics_to_file(metrics, architecture_name=architecture_name)
    
    logger.info("="*80)
    logger.info("PIPELINE CONCLUÍDO COM SUCESSO")
    logger.info("="*80)
    
    return model, metrics


def main(kwargs = None, architecture_name: str = "MobileNetV2"):

    parser = argparse.ArgumentParser(description='Treina e avalia MobileNetV2 para predição de subtipos moleculares de câncer gástrico.')
    parser.add_argument('--arch', type=str, default='MobileNetV2', help='Arquitetura a ser utilizada (MobileNetV2, ResNet50V2, VGG19, NASNetMobile, InceptionV3, DenseNet201)')
    parser.add_argument('--base_path', type=str, default='sanity_test_dataset/', help='Caminho base para os datasets (train, val, test)')
    
    # parser.add_argument('--epochs', type=int, default=10, help='Número máximo de épocas para treinamento')
    # parser.add_argument('--batch_size', type=int, default=32, help='Tamanho do batch para treinamento')
    # parser.add_argument('--base_lr', type=float, default=0.0001, help='Taxa de aprendizado inicial')
    # parser.add_argument('--phase1_ratio', type=float, default=0.4, help='Proporção de épocas para fase 1 (fine tuning com base congelada)')
    # parser.add_argument('--unfreeze_ratio', type=float, default=0.3, help='Proporção de camadas a descongelar na fase 2 (fine tuning com top layers descongeladas)')
    # parser.add_argument('--patience', type=int, default=7, help='Número de épocas sem melhora para early stopping')
    # parser.add_argument('--min_delta', type=float, default=0.001, help='Delta mínimo para considerar melhora no early stopping')
    
    args = parser.parse_args(kwargs)

    run_full_pipeline(
        architecture_name=args.arch,
        base_path=args.base_path
    )

if __name__ == "__main__":
    main()