# 📊 Módulos Principais

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

- `get_class_info(dataset)` - Extrai nomes e número de classes
- `print_environment_info()` - Mostra versões de Python, TF, CUDA, etc
- `save_metrics_to_file(metrics, arch)` - Salva resultados em arquivo
