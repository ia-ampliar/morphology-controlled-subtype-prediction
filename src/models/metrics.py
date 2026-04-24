from tensorflow.keras import backend as K

def check_units(y_true, y_pred):
    """Valida shapes de y_true e y_pred para métricas binárias."""
    if y_pred.shape[1] != 1:
        y_pred = y_pred[:, 1:2]
        y_true = y_true[:, 1:2]
    return y_true, y_pred


def precision(y_true, y_pred):
    """
    Calcula Precision para classificação multi-classe.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predictions
        
    Returns:
        Precision (Tensor)
    """
    y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def recall(y_true, y_pred):
    """
    Calcula Recall para classificação multi-classe.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predictions
        
    Returns:
        Recall (Tensor)
    """
    y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def f1(y_true, y_pred):
    """
    Calcula F1-Score para classificação multi-classe.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predictions
        
    Returns:
        F1-Score (Tensor)
    """
    def recall_local(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + K.epsilon())
    
    def precision_local(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        return true_positives / (predicted_positives + K.epsilon())
    
    y_true, y_pred = check_units(y_true, y_pred)
    prec = precision_local(y_true, y_pred)
    rec = recall_local(y_true, y_pred)
    return 2 * ((prec * rec) / (prec + rec + K.epsilon()))