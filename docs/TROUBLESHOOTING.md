
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
