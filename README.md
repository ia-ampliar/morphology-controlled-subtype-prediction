# Análise de Imagens de Histopatologia & Predição de Subtipos Moleculares

Repositório contendo notebooks Jupyter e scripts Python para aplicações de deep learning em histopatologia, com foco em **câncer gástrico**. O projeto implementa múltiplas arquiteturas de redes neurais convolucionais (CNN) modernas pré-treinadas para classificação de subtipos moleculares.

## 📋 Visão Geral do Projeto

**Predição de Subtipos Moleculares de Câncer Gástrico**:

Explora a predição de subtipos moleculares de câncer gástrico utilizando várias CNNs em imagens histopatológicas. O objetivo principal é investigar se essas predições se mantêm quando a morfologia histológica é controlada. Os modelos são avaliados quanto à estabilidade e sensibilidade a variações de taxa de aprendizado (0.0001 e 0.001) usando o otimizador Adam com early stopping implementado em duas fases de fine tuning.

### Características principais:
- ✅ **6 Arquiteturas de modelos** pré-treinados (ImageNet)
- ✅ **Pipeline automático** com treinamento em 2 fases
- ✅ **Early stopping** com paciência configurável
- ✅ **Fine tuning adaptativo** com descongelamento de camadas
- ✅ **Jupyter Notebooks** para experimentação interativa
- ✅ **CLI robusta** para execução de pipelines

## 🧠 Arquiteturas Suportadas

O repositório implementa e avalia as seguintes arquiteturas pré-treinadas:

| Arquitetura | Notebook | Comando CLI |
|------------|----------|-------------|
| MobileNetV2 | `notebooks/MobileNetV2.ipynb` | `--arch MobileNetV2` |
| DenseNet201 | `notebooks/DenseNet201.ipynb` | `--arch DenseNet201` |
| InceptionV3 | `notebooks/InceptionV3.ipynb` | `--arch InceptionV3` |
| ResNet50V2 | `notebooks/ResNet50V2.ipynb` | `--arch ResNet50V2` |
| VGG19 | `notebooks/VGG19.ipynb` | `--arch VGG19` |
| NASNetMobile | `notebooks/NASNetMobile.ipynb` | `--arch NASNetMobile` |

## 🛠️ Ambiente & Dependências

### Requisitos do Sistema:
- **Python**: 3.10.12
- **CUDA** (recomendado para GPU): 11.x ou 12.x
- **RAM mínimo**: 8GB (16GB recomendado)
- **VRAM GPU** (se disponível): 4GB+

### Dependências Python:

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

## 📦 Instalação Rápida

### 1. Preparar Ambiente Virtual

```bash
# Clonar/navegar para o repositório
cd molecular_subtype_notebooks

# Criar virtualenv
python3.10 -m venv venv

# Ativar (Linux/Mac)
source venv/bin/activate

# Ativar (Windows)
# venv\Scripts\activate
```

### 2. Instalar Dependências

```bash
# Upgrade pip
pip install --upgrade pip

# Instalar requisitos
pip install -r requirements.txt

# (Opcional) Para suporte a GPU com CUDA
pip install tensorflow[and-cuda]==2.15.0.post1
```

### 3. Preparar Dataset

Estruture seus dados nos diretórios esperados:

```
sanity_test_dataset/
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

> **Nota**: As imagens são automaticamente redimensionadas para 224×224 pixels durante o carregamento.

## 🚀 Como Usar

### Opção 1: Executar Pipeline Completo via CLI (Recomendado)

```bash
# Execução simples (MobileNetV2, configuração padrão)
python run_full_pipeline.py

# Com arquitetura específica
python run_full_pipeline.py --arch DenseNet201

# Com caminho customizado para dados
python run_full_pipeline.py --arch ResNet50V2 --base_path ./meus_dados/

# Combinado
python run_full_pipeline.py --arch InceptionV3 --base_path ./dataset_customizado/
```

#### Parâmetros disponíveis:

```
--arch {MobileNetV2, ResNet50V2, VGG19, NASNetMobile, InceptionV3, DenseNet201}
    Arquitetura do modelo a ser treinado (padrão: MobileNetV2)

--base_path <caminho>
    Caminho base onde os dados estão organizados (padrão: sanity_test_dataset/)
    Deve conter: train/, val/, test/
```

#### Exemplo completo:

```bash
python run_full_pipeline.py --arch DenseNet201 --base_path ./gastric_cancer_dataset/
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
    'epochs': 50,
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

## 📁 Estrutura do Projeto

```
molecular_subtype_notebooks/
├── README.md                      # Este arquivo
├── requirements.txt               # Dependências Python
├── run_full_pipeline.py          # Script CLI principal ⭐
│
├── notebooks/                     # Jupyter Notebooks por arquitetura
│   ├── MobileNetV2.ipynb
│   ├── DenseNet201.ipynb
│   ├── InceptionV3.ipynb
│   ├── ResNet50V2.ipynb
│   ├── VGG19.ipynb
│   └── NASNetMobile.ipynb
│
├── src/                           # Código modularizado
│   ├── models/                    # Lógica de modelos
│   │   ├── build_model.py         # Construção de modelos
│   │   ├── dataset.py             # Carregamento de dados
│   │   ├── train_model.py         # Treinamento (2 fases)
│   │   ├── inference.py           # Avaliação/Inferência
│   │   └── metrics.py             # Cálculo de métricas
│   │
│   └── utils/                     # Utilitários
│       └── utils.py               # Funções auxiliares
│
├── sanity_test_dataset/           # Dataset de teste/sanidade
│   ├── train/
│   ├── val/
│   └── test/
│
└── outputs/                       # Resultados gerados
    ├── checkpoints/               # (gitignored) Modelos salvos
    ├── metrics/                   # Logs de métricas
    └── plots/                     # Gráficos e visualizações
```

## 📊 Módulos Principais

### `run_full_pipeline.py` - Entrypoint Principal

Script CLI que orquestra todo o pipeline de treinamento e avaliação.

```python
python run_full_pipeline.py --arch MobileNetV2 --base_path ./dados/
```

**Fluxo**:
1. Carrega configurações
2. Imprime informações do ambiente
3. Carrega datasets (train/val/test)
4. Constrói modelo
5. Treina Fase 1 (base congelada)
6. Treina Fase 2 (camadas descongeladas)
7. Avalia no conjunto de teste
8. Salva métricas

### `src/models/build_model.py` - Construção de Modelos

Define como cada arquitetura é carregada e modificada:

- `get_preprocessing_function(arch)` - Retorna função de pré-processamento específica
- `get_base_model(arch, img_shape, input_tensor)` - Carrega modelo base pré-treinado
- `get_dense_layers(num_classes, basemodel, arch)` - Adiciona camadas customizadas
- `build_model(img_shape, num_classes, arch)` - Constrói modelo completo

### `src/models/dataset.py` - Carregamento de Dados

Gerencia o carregamento dos datasets:

- `setup_data_directories(config, base_path)` - Prepara caminhos
- `load_dataset(datagen, directory, batch_size)` - Carrega um dataset
- `load_all_datasets(train_dir, val_dir, test_dir, batch_size)` - Carrega todos os 3 datasets

**Augmentação**: Usa `ImageDataGenerator` do Keras (redimensionamento, normalização).

### `src/models/train_model.py` - Treinamento

Implementa treinamento em 2 fases:

- `train_phase1(model, basemodel, train_data, val_data, config, arch)` - Fase 1: Base congelada
- `train_phase2(arch, model, basemodel, train_data, val_data, config, history_phase1)` - Fase 2: Descongelamento progressivo

**Estratégia**:
- **Fase 1** (~40% das épocas): Treina apenas as camadas top (base congelada)
- **Fase 2** (~60% das épocas): Descongela ~30% das camadas base e continua treinamento

### `src/models/inference.py` - Avaliação e Inferência

Avalia o modelo treinado:

- `evaluate_model(model, test_data, class_names, arch)` - Calcula métricas no conjunto de teste
- Retorna: acurácia, precision, recall, F1, matriz de confusão

### `src/utils/utils.py` - Utilitários

Funções auxiliares:

- `get_class_info(dataset)` - Extrai nomes e número de classes
- `print_environment_info()` - Mostra versões de Python, TF, CUDA, etc
- `save_metrics_to_file(metrics, arch)` - Salva resultados em arquivo

## ⚙️ Configuração do Pipeline

As configurações padrão estão em `run_full_pipeline.py`:

```python
{
    'base_path': 'sanity_test_dataset/',
    'img_shape': (224, 224, 3),
    'batch_size': 32,
    'epochs': 5,
    'base_learning_rate': 0.0001,
    'patience': 7,  # Early stopping
    'min_delta': 0.001,  # Delta mínimo de melhora
    'phase1_ratio': 0.4,  # 40% das épocas em Fase 1
    'unfreeze_ratio': 0.3,  # Descongelar 30% das camadas em Fase 2
}
```

Para customizar, edite `load_config()` em `run_full_pipeline.py`.

## 📊 Saídas & Resultados

O pipeline gera os seguintes outputs em `outputs/`:

```
outputs/
├── metrics/
│   └── {architecture}_metrics.txt     # Métricas finais (acc, prec, recall, F1)
├── plots/
│   ├── {architecture}_confusion_matrix.png
│   ├── {architecture}_training_history.png
│   └── {architecture}_roc_curve.png
└── logs/
    └── {architecture}_training.log    # Log detalhado do treinamento
```

**Exemplo de saída**:
```
MobileNetV2_metrics.txt:
---
Accuracy: 0.8523
Precision: 0.8641
Recall: 0.8412
F1-Score: 0.8525
```

## 🔄 Fluxo de Treinamento

```
1. Setup & Carregamento
   ↓
2. Construção do Modelo
   ├─ Carregar base pré-treinada
   ├─ Congelar camadas base
   └─ Adicionar camadas densas customizadas
   ↓
3. Fase 1 de Treinamento (40% de épocas)
   ├─ Base congelada
   ├─ Treina apenas top layers
   ├─ Early stopping (patience=7)
   └─ Valida em cada época
   ↓
4. Fase 2 de Treinamento (60% de épocas)
   ├─ Descongela ~30% das camadas base
   ├─ Treina com learning rate reduzida
   ├─ Early stopping
   └─ Valida em cada época
   ↓
5. Avaliação Final
   ├─ Teste no conjunto de teste
   ├─ Calcula métricas (acc, p, r, f1)
   ├─ Gera matriz de confusão
   └─ Salva resultados
```

## 🐛 Troubleshooting

### Erro: `ModuleNotFoundError: No module named 'tensorflow'`

**Solução**:
```bash
pip install tensorflow[and-cuda]==2.15.0.post1
```

### Erro: `FileNotFoundError: [Errno 2] No such file or directory: 'sanity_test_dataset/train'`

**Solução**: Verifique se o dataset está estruturado corretamente:
```bash
# Verificar estrutura
ls -la sanity_test_dataset/
ls -la sanity_test_dataset/train/
```

Ou use um caminho customizado:
```bash
python run_full_pipeline.py --base_path ./seu_dataset/
```

### Erro: `CUDA out of memory`

**Soluções**:
1. Reduzir `batch_size` (padrão: 32 → tente 16)
2. Editar configuração em `run_full_pipeline.py`:
   ```python
   config['batch_size'] = 16
   ```
3. Usar CPU (desativar CUDA):
   ```bash
   CUDA_VISIBLE_DEVICES=-1 python run_full_pipeline.py
   ```

### Notebooks não encontram módulos de `src/`

**Solução**: Execute o notebook a partir da pasta raiz do projeto:
```bash
# Errado
cd notebooks/
jupyter notebook

# Correto
cd ..  # Volta para molecular_subtype_notebooks/
jupyter notebook notebooks/MobileNetV2.ipynb
```

### Lentidão sem GPU

**Verificar se TensorFlow detecta GPU**:
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Se não detectar**:
- Instale CUDA 11.x ou 12.x
- Reinstale TensorFlow com suporte a GPU:
  ```bash
  pip install --upgrade tensorflow[and-cuda]
  ```

### Early stopping não funciona

Verifique parâmetros em `run_full_pipeline.py`:
```python
config = {
    ...
    'patience': 7,  # Número de épocas sem melhora
    'min_delta': 0.001,  # Delta mínimo
}
```

## 📈 Resultados Esperados

Em um dataset típico de câncer gástrico com bom balanceamento:

| Métrica | Esperado |
|---------|----------|
| **Acurácia Teste** | 80-90% |
| **Precision** | 78-89% |
| **Recall** | 79-88% |
| **F1-Score** | 79-89% |

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
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [Transfer Learning Guide](https://www.tensorflow.org/guide/transfer_learning)

## 📝 Licença

MIT

---

**Dúvidas ou problemas?** Abra uma issue ou consulte a documentação do TensorFlow.