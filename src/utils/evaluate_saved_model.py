"""
Utility to evaluate a saved model on a test dataset.
"""

import os
import json
from datetime import datetime
from tensorflow.keras.models import load_model
from src.models.dataset import load_dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.models.inference import evaluate_model
from src.models.metrics import f1, precision, recall

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_saved_model(model_path: str, test_path: str, architecture_name: str = None):
    """
    Loads a saved model and evaluates it on the test dataset.

    Args:
        model_path (str): Path to the model file (.h5)
        test_path (str): Path to the test directory
        architecture_name (str): Architecture name for logs
    """
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = load_model(model_path, custom_objects={
        "f1": f1,
        "precision": precision,
        "recall": recall
    })
    print(f"Model loaded from: {model_path}")
    
    # Configure test dataset
    batch_size = 32
    
    # Create generator for test
    test_datagen = ImageDataGenerator(rescale=None)
    test_data = load_dataset(test_datagen, test_path, batch_size)

    class_names = list(test_data.class_indices.keys())
    logger.info(f"Classes detected: {class_names}")
    
    # Evaluate model
    metrics = evaluate_model(model, test_data, class_names, architecture_name)
    

if __name__ == "__main__":
    # Exemplo de uso
    model_path = "outputs/models/MobileNetV2_model_2026-04-22_14-38-11.h5"
    test_path = "sanity_test_dataset/test/"
    architecture_name = "MobileNetV2"
    
    evaluate_saved_model(model_path, test_path, architecture_name)
