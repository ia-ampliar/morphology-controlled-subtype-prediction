import os
from typing import List, Dict

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.models import Model


from src.models.metrics import precision, recall, f1

import logging


# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compile_model(model: Model, learning_rate: float = 0.0001, loss: str = "categorical_crossentropy"):
    """
    Compila o modelo com otimizador e métricas customizadas.
    
    Args:
        model (Model): Modelo Keras
        learning_rate (float): Taxa de aprendizado
    """
    opt = Adam(learning_rate=learning_rate)
    metrics = ['acc', f1, precision, recall, tf.keras.metrics.AUC()]
    
    model.compile(
        loss=loss,
        optimizer=opt,
        metrics=metrics
    )
    
    logger.info(f"Modelo compilado com learning_rate={learning_rate}")


# ============================================================================
# CONGELAMENTO/DESCONGELAMENTO DE CAMADAS
# ============================================================================

def freeze_base_model(basemodel):
    """Congela todas as camadas do modelo base."""
    for layer in basemodel.layers:
        layer.trainable = False
    logger.info("Camadas base congeladas")


def unfreeze_top_layers(basemodel, unfreeze_ratio: float = 0.3):
    """
    Descongela as camadas finais do modelo base (exceto BatchNormalization).
    
    Args:
        basemodel: Modelo base
        unfreeze_ratio (float): Proporção de camadas a descongelar (0.3 = 30%)
    """
    num_layers = len(basemodel.layers)
    num_to_unfreeze = int(num_layers * unfreeze_ratio)
    
    for layer in basemodel.layers[-num_to_unfreeze:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    
    logger.info(f"Descongeladas as {num_to_unfreeze} camadas finais (exclui BatchNormalization)")


# ============================================================================
# CALLBACKS E TREINAMENTO
# ============================================================================

def create_callbacks(checkpoint_path: str, patience: int = 7) -> List:
    """
    Cria callbacks para early stopping e model checkpoint.
    
    Args:
        checkpoint_path (str): Caminho para salvar o melhor modelo
        patience (int): Número de épocas sem melhora antes de parar
        
    Returns:
        List: Lista de callbacks
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=patience
    )
    
    model_ckpt = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True
    )
    
    return [early_stop, model_ckpt]


def train_phase1(model: Model, basemodel: Model, train_data, val_data, config: Dict, architecture_name: str) -> Dict:
    """
    Treina Fase 1 (fine tuning com base congelada).
    
    Treina apenas a camada densa por 40% das épocas totais.
    
    Args:
        model (Model): Modelo Keras
        train_data: Dataset de treino
        val_data: Dataset de validação
        config (Dict): Configuração com epochs e outras opções
        architecture_name (str): Nome da arquitetura utilizada
        
    Returns:
        Dict: History do treinamento
    """

    # congelar camadas base
    for layer in basemodel.layers:
        layer.trainable = False
    logger.info("Camadas base congeladas")

    # Definir função de perda
    loss = 'binary_crossentropy' if architecture_name == "DenseNet201" else 'categorical_crossentropy'
    
    # Compilar modelo 
    compile_model(model, config['base_learning_rate'], loss)

    initial_epochs = int(config['epochs'] * config['phase1_ratio'])
    
    logger.info(f"Fase 1: Treinando por {initial_epochs} épocas")
    
    callbacks = create_callbacks(f'callbacks/melhor_modelo_{architecture_name}_ft.hdf5')
    
    history = model.fit(
        train_data,
        epochs=initial_epochs,
        steps_per_epoch=len(train_data),
        validation_data=val_data,
        validation_steps=len(val_data),
        callbacks=callbacks
    )
    
    logger.info("Fase 1 concluída")
    return history


def train_phase2(architecture_name: str, model: Model, basemodel, train_data, val_data, config: Dict, 
                 history_phase1) -> Dict:
    """
    Treina Fase 2 (fine tuning com top layers descongeladas).
    
    Descongela as 30% camadas finais e treina com learning_rate reduzida.
    
    Args:
        architecture_name (str): Nome da arquitetura utilizada
        model (Model): Modelo Keras
        basemodel: Modelo base MobileNetV2
        train_data: Dataset de treino
        val_data: Dataset de validação
        config (Dict): Configuração
        history_phase1: History da fase 1
        
    Returns:
        Dict: History do treinamento
    """
    # Descongelar top layers
    unfreeze_top_layers(basemodel, config['unfreeze_ratio'])
    
    # Recompilar com learning rate reduzida
    new_learning_rate = config['base_learning_rate'] / 10



    compile_model(model, learning_rate=new_learning_rate,)
    
    logger.info(f"Fase 2: Treinando com learning_rate={new_learning_rate}")
    
    callbacks = create_callbacks(f'callbacks/melhor_modelo_{architecture_name}_ft.hdf5')
    
    # Continuar do époco anterior
    initial_epoch = history_phase1.epoch[-1]
    history_fine = model.fit(
        train_data,
        epochs=config['epochs'],
        initial_epoch=initial_epoch,
        steps_per_epoch=len(train_data),
        validation_data=val_data,
        validation_steps=len(val_data),
        callbacks=callbacks
    )
    
    logger.info("Fase 2 concluída")
    return history_fine