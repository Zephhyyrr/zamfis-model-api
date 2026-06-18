import joblib
import os

def load_models():
    """
    Memuat model hibrida untuk income dan expense
    
    Returns:
        dict: Dictionary berisi model income dan expense
    """
    models = {}
    
    # Load Model Income (Uang Masuk)
    income_path = 'models/model_surau_uang_masuk.pkl'
    if os.path.exists(income_path):
        m = joblib.load(income_path)
        models['income'] = m
    else:
        print(f"Warning: Model income tidak ditemukan di {income_path}")
    
    # Load Model Expense (Uang Keluar)
    expense_path = 'models/model_surau_uang_keluar.pkl'
    if os.path.exists(expense_path):
        m = joblib.load(expense_path)
        models['expense'] = m
    else:
        print(f"Warning: Model expense tidak ditemukan di {expense_path}")
        
    return models
