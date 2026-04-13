# Histopathology Image Analysis & Molecular Subtype Prediction

This repository contains a collection of Jupyter Notebooks dedicated to deep learning applications in histopathology, specifically focusing on gastric cancer. The repository has various Convolutional Neural Network (CNN) architectures for molecular subtype classification.

## 📋 Project Overview

**Gastric Cancer Molecular Subtype Prediction**:
Exploring the prediction of gastric cancer molecular subtypes using various CNNs on histopathological images. A core objective is investigating whether these predictions hold when histological morphology is controlled. The models are evaluated for stability and sensitivity to learning rate variations (0.0001 and 0.001) using the Adam optimizer with early stopping.

## 🧠 Models & Notebooks

The repository explores multiple pre-trained deep learning architectures. Each notebook runs the experimental pipeline on a specific model:

- `DenseNet201.ipynb`
- `InceptionV3.ipynb`
- `MobileNetV2.ipynb`
- `NASNetMobile.ipynb`
- `ResNet50V2.ipynb`
- `VGG19.ipynb`

## 🛠️ Requirements

The notebooks were developed and tested using the following environment:

- `python=3.10.12`
- `tensorflow=2.15.0`
- `numpy=1.25.2`
- `pandas=2.0.3`
- `seaborn=0.12.2`
- `matplotlib=3.7.1`
- `opencv-python=4.8.0.76`
- `scikit-learn=1.2.2`

## 📊 Datasets

- **Gastric Cancer**: Morphology-controlled histopathology datasets are processed locally within the notebooks.

## 🚀 Usage

To run the notebooks:

1. Clone this repository.
2. Ensure you have the required dependencies installed (we recommend setting up a virtual environment).
3. Update the dataset paths in the notebooks to point to your local directories or Google Drive (the notebooks contain cells for mounting Google Drive).
4. Run the notebooks via Jupyter or Google Colab.