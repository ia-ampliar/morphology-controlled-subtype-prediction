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
   ├─ Calcula métricas (acc, p, r, f1, auc)
   ├─ Gera matriz de confusão
   └─ Salva resultados
```