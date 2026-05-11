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
└── outputs/                       # Resultados gerados (organizados por learning rate)
    ├── 0.0001/
    │   ├── dataframes/            # CSVs com predições
    │   │   └── {arch}_0.0001_test_predictions_{timestamp}.csv
    │   ├── metrics/               # JSONs com métricas
    │   │   └── {arch}_0.0001_test_metrics_{timestamp}.json
    │   └── models/                # Modelos treinados
    │       └── {arch}_0.0001_model_{timestamp}.h5
    └── 0.001/                     # Outra taxa de aprendizado
        ├── dataframes/
        ├── metrics/
        └── models/
```

**Principais mudanças**:
- Outputs agora organizados em subpastas por learning rate (ex: `outputs/0.0001/`)
- Nomes de arquivos incluem a taxa de aprendizado (ex: `MobileNetV2_0.0001_test_metrics_{timestamp}.json`)
- Suporta múltiplas taxa de aprendizado simultaneamente

## � Configuração de Sementes Aleatórias

O projeto implementa sementes aleatórias para garantir **reprodutibilidade completa** de resultados:

### Sementes Configuradas

```python
# Em src/utils/utils.py -> set_random_seeds(seed=42)
np.random.seed(42)              # NumPy
tf.random.set_seed(42)          # TensorFlow
random.seed(42)                 # Python random
tf.keras.utils.set_random_seed(42)  # Keras
```

### Onde São Utilisadas

| Componente | Arquivo | Uso |
|-----------|---------|-----|
| **Inicialização** | `run_full_pipeline.py` | Chamada no início de `main()` |
| **Dados** | `src/models/dataset.py` | Embaralhamento (shuffle) dos batches com `seed=42` |
| **Pesos** | `src/models/build_model.py` | Inicialização determinística das camadas |
| **Predições** | `src/models/inference.py` | Seed configurada antes de `evaluate_model()` |

### Impacto

- ✅ **CPU**: 100% reprodutibilidade
- ✅ **GPU**: ~95-99% reprodutibilidade (variações mínimas em FusedBatchNorm)

### Verificação de Reprodutibilidade

```bash
# Executar duas vezes e comparar resultados
python run_full_pipeline.py --arch MobileNetV2 --epochs 10
# Anotar: outputs/0.0001/metrics/MobileNetV2_*_metrics_*.json

python run_full_pipeline.py --arch MobileNetV2 --epochs 10
# Métricas devem ser idênticas
```

## 🔄 Fluxo de Treinamento

```
1. Setup & Sementes Aleatórias
   ├─ Configura set_random_seeds(42)
   └─ Garantir reprodutibilidade
   ↓
2. Carregamento de Dados
   ├─ Embaralha com seed=42
   └─ Datasets (train/val/test)
   ↓
3. Construção do Modelo
   ├─ Carregar base pré-treinada
   ├─ Congelar camadas base
   └─ Adicionar camadas densas customizadas
   ↓
4. Fase 1 de Treinamento (40% de épocas)
   ├─ Base congelada
   ├─ Treina apenas top layers
   ├─ Early stopping (patience=7)
   └─ Valida em cada época
   ↓
5. Fase 2 de Treinamento (60% de épocas)
   ├─ Descongela ~30% das camadas base
   ├─ Treina com learning rate reduzida
   ├─ Early stopping
   └─ Valida em cada época
   ↓
6. Avaliação Final
   ├─ Teste no conjunto de teste
   ├─ Calcula métricas (acc, p, r, f1, auc)
   ├─ Gera matriz de confusão
   └─ Salva resultados
```