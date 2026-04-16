from typing import Tuple

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import MobileNetV2, NASNetMobile, ResNet50V2, VGG19, InceptionV3, DenseNet201

import logging


# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_preprocessing_function(architecture_name: str):
    """
    Retorna função de pré-processamento de acordo com a arquitetura.
    
    Args:
        architecture_name (str): Nome da arquitetura (MobileNetV2, ResNet50V2, etc)

    Returns:
        function: Função de pré-processamento
    """
    if architecture_name == "MobileNetV2":
        return tf.keras.applications.mobilenet_v2.preprocess_input
    elif architecture_name == "ResNet50V2":
        return tf.keras.applications.resnet_v2.preprocess_input
    elif architecture_name == "VGG19":
        return tf.keras.applications.vgg19.preprocess_input
    elif architecture_name == "NASNetMobile":
        return tf.keras.applications.nasnet.preprocess_input
    elif architecture_name == "InceptionV3":
        return tf.keras.applications.inception_v3.preprocess_input
    elif architecture_name == "DenseNet201":
        return tf.keras.applications.densenet.preprocess_input
    else:
        raise ValueError(f"Arquitetura {architecture_name} não suportada.")


def get_base_model(architecture_name: str, img_shape: Tuple, xi: tf.Tensor) -> Model:
    """
    Retorna modelo base pré-treinado de acordo com o nome da arquitetura.
    
    Args:
        architecture_name (str): Nome da arquitetura (MobileNetV2, ResNet50V2, etc)
        img_shape (Tuple): Formato das imagens (224, 224, 3)
        xi: Imagem pré-processada

    Returns:
        Model: Modelo base pré-treinado
    """
    if architecture_name == "MobileNetV2":
        return MobileNetV2(include_top=False, weights='imagenet', input_tensor = xi, input_shape=img_shape, pooling='max')
    elif architecture_name == "ResNet50V2":
        return ResNet50V2(include_top=False, weights='imagenet', input_tensor = xi, input_shape=img_shape, pooling='max')
    elif architecture_name == "VGG19":
        return VGG19(include_top=False, weights='imagenet', input_tensor = xi, input_shape=img_shape, pooling='max')
    elif architecture_name == "NASNetMobile":
        return NASNetMobile(include_top=False, weights='imagenet', input_tensor = xi, input_shape=img_shape, pooling='max')
    elif architecture_name == "InceptionV3":
        return InceptionV3(include_top=False, weights='imagenet', input_tensor = xi, input_shape=img_shape, pooling='max')
    elif architecture_name == "DenseNet201":
        return DenseNet201(include_top=False, weights='imagenet', input_tensor = xi, input_shape=img_shape, pooling='max')
    # Adicione mais condições para outras arquiteturas conforme necessário
    else:
        raise ValueError(f"Arquitetura {architecture_name} não suportada.")
    

def get_dense_layers(num_classes: int, basemodel: Model, architecture_name: str):
    """
    Adiciona camadas densas customizadas ao modelo.
    
    Args:
        x: Saída do modelo base
        num_classes (int): Número de classes
        
    Returns:
        Tensor: Saída final com camadas densas
    """

    if architecture_name == "DenseNet201":
        x = Dense(256, activation="relu")(basemodel.output)
        x = Dropout(0.2)(x)

        x = BatchNormalization()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.2)(x)

        x = BatchNormalization()(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.2)(x)

        x = BatchNormalization()(x)
        x = Dense(32, activation="relu")(x)
        x = Dropout(0.2)(x)

    else:
        x = BatchNormalization()(basemodel.output)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.2)(x)
        
        x = BatchNormalization()(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.2)(x)

    outputs = Dense(num_classes, activation='softmax')(x)
    
    return outputs

def build_model(img_shape: Tuple, num_classes: int, architecture_name: str = "MobileNetV2") -> Model:
    """
    Constrói arquitetura {architecture_name} com camadas densas customizadas.

    Args:
        img_shape (Tuple): Formato das imagens (224, 224, 3)
        num_classes (int): Número de classes
        architecture_name (str): Nome da arquitetura utilizada

    Returns:
        Model: Modelo Keras compilado
    """
    # Input
    inputs = Input(shape=img_shape)
    
    # Pré-processamento
    xi = tf.cast(inputs, tf.float32)
    xi = get_preprocessing_function(architecture_name)(xi)

    # Base model (MobileNetV2 pré-treinado)
    basemodel = get_base_model(architecture_name, img_shape, xi)
    
    # Camadas densas
    outputs = get_dense_layers(num_classes, basemodel, architecture_name)
    
    model = Model(inputs, outputs)
    
    logger.info(f"Modelo construído com {len(model.layers)} camadas")
    logger.info(f"Número de camadas base: {len(basemodel.layers)}")
    
    return model, basemodel