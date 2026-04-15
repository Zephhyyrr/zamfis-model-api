import joblib
import os


def load_models():
    """
    Memuat kedua model (Prophet dan LightGBM Residual)
    
    Returns:
        dict: Dictionary berisi prophet_model dan lgbm_model
    """
    models = {}
    
    # Load Prophet Model
    prophet_path = 'models/prophet_model.joblib'
    if os.path.exists(prophet_path):
        models['prophet'] = joblib.load(prophet_path)
    else:
        raise FileNotFoundError(f"Prophet model tidak ditemukan: {prophet_path}")
    
    # Load LightGBM Residual Model
    lgbm_path = 'models/lgbm_residual_model.joblib'
    if os.path.exists(lgbm_path):
        models['lgbm'] = joblib.load(lgbm_path)
    else:
        raise FileNotFoundError(f"LightGBM model tidak ditemukan: {lgbm_path}")
    
    return models


def load_model(model_path='models/hybrid_model.pkl'):
    """Legacy function - gunakan load_models() untuk hybrid predictions"""
    model = joblib.load(model_path)
    return model
