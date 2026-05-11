# 📊 Módulos Principais

### `run_full_pipeline.py` - Entrypoint Principal

Script CLI que orquestra todo o pipeline de treinamento e avaliação.

```python
# Treinar uma arquitetura específica
python run_full_pipeline.py --arch MobileNetV2 --base_path ./dados/

# Treinar todas as 6 arquiteturas
python run_full_pipeline.py --all --output_dir ./resultados/

# Treinar com taxa de aprendizado e diretório customizado
python run_full_pipeline.py --arch DenseNet201 --base_lr 0.001 --output_dir ./outputs_lr001/
```

**Parâmetros principais**:
- `--arch`: Arquitetura específica (padrão: MobileNetV2)
- `--all`: Executa pipeline para todas as 6 arquiteturas
- `--base_path`: Caminho para dataset (padrão: sanity_test_dataset/)
- `--base_lr`: Taxa de aprendizado (padrão: 0.0001)
- `--output_dir`: Diretório de outputs (padrão: outputs/)
- `--epochs`: Número de épocas (padrão: 5)

**Fluxo**:
1. Configura sementes aleatórias para reprodutibilidade (`set_random_seeds()`)
2. Carrega configurações
3. Imprime informações do ambiente
4. Carrega datasets (train/val/test)
5. Constrói modelo
6. Treina Fase 1 (base congelada)
7. Treina Fase 2 (camadas descongeladas)
8. Avalia no conjunto de teste
9. Salva métricas com learning rate no nome do arquivo
10. Organiza outputs em `{output_dir}/{learning_rate}/`

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

Usa `ImageDataGenerator` do Keras.

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
- Retorna: acurácia, precision, recall, F1, AUC e matriz de confusão

### `src/utils/utils.py` - Utilitários

Funções auxiliares:

- `set_random_seeds(seed=42)` - **Configura sementes para reprodutibilidade** 🔐
  - Define seeds para NumPy, TensorFlow, Python random e Keras
  - Chamada automaticamente no início de `run_full_pipeline.py`
  - Garante resultados idênticos em execuções subsequentes
  
- `get_class_info(dataset)` - Extrai nomes e número de classes
- `print_environment_info()` - Mostra versões de Python, TF, CUDA, etc
- `setup_gpu()` - Verifica e configura GPU se disponível
- `save_metrics_to_file(metrics, arch)` - Salva resultados em arquivo
