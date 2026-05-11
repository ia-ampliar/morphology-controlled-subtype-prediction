import json
import logging
from typing import Tuple, List, Dict
import os
from datetime import datetime
import tensorflow as tf
import platform
import numpy as np
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42):
    """
    Define sementes aleatórias para garantir reprodutibilidade completa.
    
    Configura sementes para:
    - NumPy (operações matemáticas)
    - TensorFlow/Keras (inicialização de pesos, dropout, shuffling)
    - Python random (outras operações aleatórias)
    
    NOTA: tf.config.experimental.enable_op_determinism() foi removido pois não é 
    suportado com BatchNormalization em GPU. As sementes abaixo já garantem 
    reprodutibilidade em CPU e maior consistência em GPU.
    
    Args:
        seed (int): Valor da semente (padrão: 42)
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    
    logger.info(f"Sementes aleatórias configuradas com seed={seed}")


def setup_gpu():
    """
    Verifica e configura GPU se disponível.
    
    Returns:
        bool: True se GPU está disponível, False caso contrário
    """
    if not tf.test.gpu_device_name():
        logger.warning('GPU não encontrada. Usando CPU.')
        return False
    else:
        logger.info(f'Dispositivo GPU padrão: {tf.test.gpu_device_name()}')
        logger.info(f"Número de GPUs Disponíveis: {len(tf.config.list_physical_devices('GPU'))}")
        return True


def print_environment_info():
    """Imprime informações do ambiente (Python, TensorFlow, GPU)."""
    logger.info(f"Python - Versão: {platform.python_version()}")
    logger.info(f"TensorFlow - Versão: {tf.__version__}")
    setup_gpu()

def get_class_info(val_data) -> Tuple[List[str], int]:
    """
    Extrai informações de classes do dataset de validação.
    
    Args:
        val_data (DirectoryIterator): Iterator de validação
        
    Returns:
        Tuple[List[str], int]: (class_names, num_classes)
    """
    class_names = list(val_data.class_indices.keys())
    num_classes = len(class_names)
    
    logger.info(f"Classes: {class_names}")
    logger.info(f"Número de classes: {num_classes}")
    
    return class_names, num_classes


def save_metrics_to_file(metrics: Dict, output_path: str = None, architecture_name: str = None, config: Dict = None):
    """
    Salva métricas em arquivo texto.
    
    Args:
        metrics (Dict): Dicionário com métricas
        output_path (str): Caminho de saída (default: with timestamp)
        architecture_name (str): Nome da arquitetura
        config (Dict): Configuração
    """
    if output_path is None:
        output_path = f'{config["output_dir"]}/metrics/{architecture_name}_{config["base_learning_rate"]}_test_metrics_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json'
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f)
    
    logger.info(f"Métricas salvas em: {output_path}")