# Análise de Imagens de Histopatologia & Predição de Subtipos Moleculares

Repositório contendo notebooks Jupyter e scripts Python para aplicações de deep learning em histopatologia, com foco em **câncer gástrico**. O projeto implementa múltiplas arquiteturas de redes neurais convolucionais (CNN) modernas pré-treinadas para classificação de subtipos moleculares.

## Indice
- [Visão Geral do Projeto](#📋-Visão-Geral-do-Projeto)
- [Arquiteturas Suportadas](#🧠-Arquiteturas-Suportadas)
- [Ambiente & Dependências](#🛠️-Ambiente-&-Dependências)
- [Instalação Rápida](#📦-Instalação-Rápida)
- [Como Usar](#🚀-Como-Usar)
- [Saídas & Resultados](#📊-Saídas-&-Resultados)
- [Reprodutibilidade & Sementes Aleatórias](#🔐-Reprodutibilidade--Sementes-Aleatórias)
- [Fluxo de Treinamento](#🔄-Fluxo-de-Treinamento)
- [Material Suplementar](#📚-Material-Suplementar)
    - [Troubleshooting](#🐛-Troubleshooting)
    - [Módulos Principais](#📊-Módulos-Principais)
    - [Estrutura do Projeto](#📁-Estrutura-do-Projeto)
    - [Fluxo de Treinamento](#🔄-Fluxo-de-Treinamento)
- [Resultados Esperados](#📈-Resultados-Esperados)
- [Contribuições](#🤝-Contribuições)
- [Referências & Docs](#📚-Referências-&-Docs)
- [Licença](#📝-Licença)


## 📋 Project Overview

**Gastric Cancer Molecular Subtype Prediction**:

Explores the prediction of gastric cancer molecular subtypes using multiple CNNs on histopathological images. The main objective is to investigate whether these predictions hold when histological morphology is controlled. Models are evaluated for stability and sensitivity to variations in learning rates (0.0001 and 0.001) using the Adam optimizer with early stopping implemented in two fine-tuning phases.

### Main Features:
- ✅ **6 Pre-trained Model Architectures** (ImageNet)
- ✅ **Automatic Pipeline** with 2-phase training
- ✅ **Early Stopping** with configurable patience
- ✅ **Adaptive Fine Tuning** with layer unfreezing
- ✅ **Jupyter Notebooks** for interactive experimentation
- ✅ **Robust CLI** for pipeline execution
- ✅ **Reproducible Results** with fixed random seeds

## 🧠 Supported Architectures

The repository implements and evaluates the following pre-trained architectures:

| Architecture | Notebook | CLI Command |
|------------|----------|-------------|
| MobileNetV2 | `notebooks/MobileNetV2.ipynb` | `--arch MobileNetV2` |
| DenseNet201 | `notebooks/DenseNet201.ipynb` | `--arch DenseNet201` |
| InceptionV3 | `notebooks/InceptionV3.ipynb` | `--arch InceptionV3` |
| ResNet50V2 | `notebooks/ResNet50V2.ipynb` | `--arch ResNet50V2` |
| VGG19 | `notebooks/VGG19.ipynb` | `--arch VGG19` |
| NASNetMobile | `notebooks/NASNetMobile.ipynb` | `--arch NASNetMobile` |

## 🖨️ Environment & Dependencies

### System Requirements:
- **Python**: 3.10.12
- **CUDA** (recommended for GPU): 11.x or 12.x
- **Minimum RAM**: 8GB (16GB recommended)
- **GPU VRAM** (if available): 4GB+

### Python Dependencies:

```
tensorflow[and-cuda]==2.15.0.post1
numpy==1.25.2
pandas==2.0.3
seaborn==0.12.2
matplotlib==3.7.1
opencv-python==4.8.0.76
scikit-learn==1.2.2
jupyter>=1.0.0
```

## 📦 Quick Installation

### Option 1: Using Virtualenv (Python)

#### 1. Prepare Virtual Environment

```bash
# Create virtualenv
python3.10 -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

#### 2. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt --extra-index-url https://pypi.nvidia.com

```

### Option 2: Using Conda

#### 1. Create Conda Environment

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate molsub-env
```

#### 2. Install Dependencies (if necessary)

If there are dependencies that were not installed via `environment.yml`, install via pip:

```bash
pip install -r requirements.txt --extra-index-url https://pypi.nvidia.com
```

### 3. Prepare Dataset

Structure your data in the expected directories:

```
molecular_subtype_dataset/
├── train/
│   ├── class_1/
│   │   ├── image_1.jpg
│   │   └── ...
│   └── class_2/
│       ├── image_1.jpg
│       └── ...
├── val/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
```

> **Note**: Images are automatically resized to 224×224 pixels during loading.

 ## 🚀 Como Usar

### Opção 1: Executar Pipeline Completo via CLI (Recomendado)

```bash
# Execução simples (MobileNetV2, configuração padrão)
python run_full_pipeline.py

# Com arquitetura específica
python run_full_pipeline.py --arch DenseNet201

# Com caminho customizado para dados
python run_full_pipeline.py --arch ResNet50V2 --base_path ./meus_dados/

# Rodar pipeline em todas as arquiteturas (DenseNet201, InceptionV3, MobileNetV2, NASNetMobile, ResNet50V2, VGG19)
python run_full_pipeline.py --all

# Combinado com diretório de saída customizado
python run_full_pipeline.py --arch InceptionV3 --base_path ./dataset_customizado/ --output_dir ./resultados/

# Treinar todas as arquiteturas com taxa de aprendizado diferente
python run_full_pipeline.py --all --base_lr 0.001 --output_dir ./outputs_lr001/
```

#### Parâmetros disponíveis:

```
--arch {MobileNetV2, ResNet50V2, VGG19, NASNetMobile, InceptionV3, DenseNet201}
    Arquitetura do modelo a ser treinado (padrão: MobileNetV2)
    Ignorado quando --all é utilizado

--all
    Executa o pipeline completo para todas as 6 arquiteturas
    Usa os mesmos parâmetros (base_path, epochs, base_lr, output_dir) para cada uma

--base_path <caminho>
    Caminho base onde os dados estão organizados (padrão: sanity_test_dataset/)
    Deve conter: train/, val/, test/

--epochs <número>
    Número máximo de épocas para treinamento (padrão: 5)

--base_lr <valor>
    Taxa de aprendizado inicial (padrão: 0.0001)

--output_dir <caminho>
    Diretório para salvar modelos, métricas e visualizações (padrão: outputs/)
    Os arquivos serão organizados em subpastas de acordo com a taxa de aprendizado (ex: outputs/0.0001/)
```

### Opção 2: Executar Notebooks Interativos via Jupyter

```bash
# Iniciar Jupyter Lab (recomendado)
jupyter lab

# Ou Jupyter Notebook
jupyter notebook
```

Navegue até `notebooks/` e abra o notebook desejado:
- `MobileNetV2.ipynb`
- `DenseNet201.ipynb`
- `InceptionV3.ipynb`
- Etc...

**Vantagens dos notebooks**:
- Visualização passo a passo do pipeline
- Gráficos e métricas em tempo real
- Células customizáveis para experimentação
- Fácil visualização de resultados intermediários

### Opção 3: Usar Como Módulo Python

```python
from src.models.dataset import setup_data_directories, load_all_datasets
from src.models.build_model import build_model
from src.models.train_model import train_phase1, train_phase2
from src.models.inference import evaluate_model
from src.utils.utils import save_metrics_to_file

# Configurar
config = {
    'base_path': 'sanity_test_dataset/',
    'img_shape': (224, 224, 3),
    'batch_size': 32,
    'epochs': 100,
    'base_learning_rate': 0.0001,
    'patience': 7,
    'min_delta': 0.001,
    'phase1_ratio': 0.4,
    'unfreeze_ratio': 0.3,
}

# Carregar dados
train_dir, val_dir, test_dir = setup_data_directories(config, config['base_path'])
train_data, val_data, test_data = load_all_datasets(
    train_dir, val_dir, test_dir, config['batch_size']
)

# Construir modelo
model, basemodel = build_model(config['img_shape'], num_classes=2, architecture_name='MobileNetV2')

# Treinar (Fase 1)
history_phase1 = train_phase1(model, basemodel, train_data, val_data, config, 'MobileNetV2')

# Treinar (Fase 2)
history_phase2 = train_phase2('MobileNetV2', model, basemodel, train_data, val_data, config, history_phase1)

# Avaliar
metrics = evaluate_model(model, test_data, class_names, architecture_name='MobileNetV2')
```

## 📊 Saídas & Resultados

O pipeline gera outputs organizados por taxa de aprendizado em `outputs/{learning_rate}/`:

```
outputs/
├── 0.0001/                                               # Subpasta por taxa de aprendizado
│   ├── dataframes/
│   │   ├── MobileNetV2_0.0001_test_predictions_{timestamp}.csv
│   │   ├── DenseNet201_0.0001_test_predictions_{timestamp}.csv
│   │   └── ...
│   ├── metrics/
│   │   ├── MobileNetV2_0.0001_test_metrics_{timestamp}.json
│   │   ├── DenseNet201_0.0001_test_metrics_{timestamp}.json
│   │   └── ...
│   └── models/
│       ├── MobileNetV2_0.0001_model_{timestamp}.h5
│       ├── DenseNet201_0.0001_model_{timestamp}.h5
│       └── ...
├── 0.001/                                                # Outra taxa de aprendizado
│   ├── dataframes/
│   ├── metrics/
│   └── models/
└── ...
```

**Estrutura de nomes dos arquivos**:
- Formato: `{architecture}_{learning_rate}_{type}_{timestamp}.{ext}`
- Exemplo: `MobileNetV2_0.0001_test_predictions_2026-05-08_08-27-24.csv`
- Inclui a taxa de aprendizado no nome para fácil identificação

**Exemplo de saída JSON** (`MobileNetV2_0.0001_test_metrics_2026-05-08_08-27-24.json`):
```json
{
  "loss": 4.176743507385254,
  "accuracy": 0.5,
  "f1_score": 0.0,
  "precision": 0.0,
  "recall": 0.0,
  "auc": 0.550000011920929
}
```



## � Reprodutibilidade & Sementes Aleatórias

### Garantindo Resultados Reproduzíveis

O pipeline está configurado com **sementes aleatórias** para garantir que você obtenha os mesmos resultados em execuções subsequentes.

### Como Funciona

As sementes são automaticamente configuradas no **início de cada execução** através da função `set_random_seeds(seed=42)` que define:

```python
# Sementes configuradas automaticamente:
- NumPy random seed (operações matemáticas)
- TensorFlow random seed (inicialização de pesos)
- Python random seed (outras operações aleatórias)
- Keras/tf.keras seed (dropout, shuffling de batches)
```

### Customizar Seed

Para alterar a seed (não recomendado para reprodutibilidade padrão):

```python
# Editar em src/utils/utils.py
def set_random_seeds(seed: int = 42):  # Mudar 42 para outro valor
    # ...
```

### Verificar Reprodutibilidade

Execute o mesmo comando duas vezes e compare os resultados:

```bash
# Primeira execução
python run_full_pipeline.py --arch MobileNetV2 --epochs 10
ls outputs/0.0001/metrics/MobileNetV2_*.json

# Segunda execução
python run_full_pipeline.py --arch MobileNetV2 --epochs 10
ls outputs/0.0001/metrics/MobileNetV2_*.json

# Os arquivos JSON devem ter métricas idênticas ou muito aproximadas
```

## �📚 Material Suplementar

- ### 🐛 Troubleshooting
- ### 📊 Módulos Principais 
- ## 📁 Estrutura do Projeto
- ## 🔄 Fluxo de Treinamento

## 📈 Resultados Esperados

Em um dataset típico de câncer gástrico com bom balanceamento:

| Métrica | Esperado |
|---------|----------|
| **Acurácia** | 52-71% |
| **Precision** | 54-74% |
| **Recall** | 51-79% |
| **F1-Score** | 57-73% |
| **AUC** | 54-75%|

*Os resultados variam conforme o dataset, número de classes e balanceamento.*

## 🤝 Contribuições

Para adicionar novas arquiteturas:

1. Edite `src/models/build_model.py`:
   - Adicione função `get_preprocessing_function()`
   - Adicione modelo em `get_base_model()`
   - Configure camadas denasas em `get_dense_layers()`

2. Crie um novo notebook em `notebooks/{ArchName}.ipynb`

3. Atualize a tabela de arquiteturas neste README

## 📚 Referências & Docs

- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [Keras Applications](https://keras.io/api/applications/)
- [Transfer Learning Guide](https://www.tensorflow.org/guide/transfer_learning)

## 📝 Licença

MIT

---

**Dúvidas ou problemas?** Abra uma issue ou consulte a documentação do TensorFlow.