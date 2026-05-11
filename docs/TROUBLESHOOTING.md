
# 🐛 Troubleshooting

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

### Erros ao usar `--all` para treinar múltiplas arquiteturas

**Erro**: Alguns modelos treinam mas outros falham

**Solução**: Verifique se:
1. Todos os datasets (train/val/test) estão disponíveis
2. Há memória GPU suficiente:
   ```bash
   CUDA_VISIBLE_DEVICES=-1 python run_full_pipeline.py --all  # Força CPU
   ```
3. Tente treinar cada arquitetura individualmente para identificar qual falha:
   ```bash
   python run_full_pipeline.py --arch MobileNetV2
   python run_full_pipeline.py --arch DenseNet201
   # etc...
   ```

### Outputs não aparecem no diretório esperado

**Erro**: Arquivos salvos em lugar diferente do esperado

**Solução**: O `--output_dir` define o diretório base. Outputs serão organizados em subpastas por taxa de aprendizado:
```bash
# Exemplo: --output_dir ./resultados/ + --base_lr 0.0001
# Outputs vão para: ./resultados/0.0001/

python run_full_pipeline.py --output_dir ./meus_resultados/ --base_lr 0.001
# Verificar: ls -la ./meus_resultados/0.001/
```

### Arquivos de modelo salvos com nomes diferentes

Os arquivos agora incluem a taxa de aprendizado no nome para fácil identificação:
```
MobileNetV2_0.0001_model_{timestamp}.h5
DenseNet201_0.001_model_{timestamp}.h5
```

Se quiser encontrar modelos de uma taxa de aprendizado específica:
```bash
# Modelos de learning rate 0.0001
ls -la outputs/0.0001/models/

# Modelos de learning rate 0.001  
ls -la outputs/0.001/models/
```

### Resultados diferentes em cada execução (Não-Reprodutibilidade)

**Problema**: Os resultados variam significativamente entre execuções do mesmo comando.

**Solução**: As sementes aleatórias já estão configuradas por padrão (`seed=42`). Se ainda assim obtiver diferenças:

1. **Verificar se set_random_seeds() está sendo chamada**:
   ```bash
   # Verificar mensagem de log
   python run_full_pipeline.py --arch MobileNetV2 2>&1 | grep "Sementes"
   # Deve mostrar: "[INFO] ... Sementes aleatórias configuradas com seed=42"
   ```

2. **Tentar em GPU vs CPU**:
   - Variações pequenas (~1-2%) em GPU são normais (operações paralelas não-determinísticas)
   - CPU deve dar 100% reprodutibilidade:
   ```bash
   CUDA_VISIBLE_DEVICES=-1 python run_full_pipeline.py --arch MobileNetV2
   ```

3. **Executar duas vezes e comparar**:
   ```bash
   # Primeira execução
   python run_full_pipeline.py --arch MobileNetV2 --epochs 5
   python -c "import json; print(json.load(open('outputs/0.0001/metrics/MobileNetV2_*_metrics_*.json'.replace('*', 'temp'))))"
   
   # Segunda execução (deve ter valores idênticos)
   python run_full_pipeline.py --arch MobileNetV2 --epochs 5
   ```

4. **Verificar seed em src/utils/utils.py**:
   ```python
   # Confirmar que seed=42 está definida
   def set_random_seeds(seed: int = 42):
       # ...
   ```

**Nota**: Veja a seção "Reprodutibilidade & Sementes Aleatórias" no README.md para mais detalhes.
