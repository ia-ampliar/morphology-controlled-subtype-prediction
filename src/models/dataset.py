import os
from typing import Dict, Tuple
import logging
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_data_directories(config: Dict, base_path: str) -> Tuple[str, str, str]:
    """
    Prepara caminhos dos diretórios de dados.
    
    Args:
        config (dict): Configuração com base_path
        base_path (str): Caminho base para os datasets
        
    Returns:
        Tuple[str, str, str]: Caminhos (train_dir, val_dir, test_dir)
    """
    
    train_dir = os.path.join(base_path, 'train')
    val_dir = os.path.join(base_path, 'val')
    test_dir = os.path.join(base_path, 'test')
    
    for directory in [train_dir, val_dir, test_dir]:
        if not os.path.exists(directory):
            logger.warning(f"[ERROR] - Diretório não encontrado: {directory}")
    
    return train_dir, val_dir, test_dir


def load_dataset(datagen: ImageDataGenerator, directory: str, 
                 batch_size: int = 32) -> tf.keras.preprocessing.image.DirectoryIterator:
    """
    Carrega dataset de um diretório usando ImageDataGenerator.
    
    Args:
        datagen (ImageDataGenerator): Gerador configurado
        directory (str): Caminho do diretório
        batch_size (int): Tamanho do batch
        
    Returns:
        DirectoryIterator: Iterator com dados carregados
    """
    return datagen.flow_from_directory(
        directory,
        batch_size=batch_size,
        target_size=(224, 224),
        class_mode="categorical"
    )


def load_all_datasets(train_dir: str, val_dir: str, test_dir: str, 
                      batch_size: int = 32) -> Tuple:
    """
    Carrega todos os datasets (train, val, test).
    
    Args:
        train_dir (str): Caminho do dataset de treino
        val_dir (str): Caminho do dataset de validação
        test_dir (str): Caminho do dataset de teste
        batch_size (int): Tamanho do batch
        
    Returns:
        Tuple: (train_data, val_data, test_data)
    """

    train_datagen = ImageDataGenerator(rescale=None)
    valid_datagen = ImageDataGenerator(rescale=None)
    test_datagen = ImageDataGenerator(rescale=None)
    
    train_data = load_dataset(train_datagen, train_dir, batch_size)
    val_data = load_dataset(valid_datagen, val_dir, batch_size)
    test_data = load_dataset(test_datagen, test_dir, batch_size)
    
    logger.info(f"[INFO] - Datasets carregados: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    
    return train_data, val_data, test_data