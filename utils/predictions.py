import pandas as pd
import numpy as np
from hijri_converter import Gregorian


def get_hijri_month(gregorian_date):
    """
    Mengkonversi tanggal Gregorian ke bulan Hijriah
    
    Args:
        gregorian_date (datetime): Tanggal Gregorian
    
    Returns:
        int: Bulan Hijriah (1-12)
    """
    try:
        hijri = Gregorian(
            gregorian_date.year, 
            gregorian_date.month, 
            gregorian_date.day
        ).to_hijri()
        return hijri.month
    except Exception:
        return 0  # Default jika konversi gagal


def get_hijri_info(gregorian_date):
    """
    Mengkonversi tanggal Gregorian ke bulan dan hari Hijriah
    
    Args:
        gregorian_date (datetime): Tanggal Gregorian
    
    Returns:
        tuple: (bulan_hijriah, hari_hijriah)
    """
    try:
        hijri = Gregorian(
            gregorian_date.year, 
            gregorian_date.month, 
            gregorian_date.day
        ).to_hijri()
        return hijri.month, hijri.day
    except Exception:
        return 0, 0  # Default jika konversi gagal


def make_predictions(models, start_date, end_date):
    """
    Melakukan prediksi hybrid donasi menggunakan:
    1. Prophet: Menangkap trend dan seasonality
    2. LightGBM: Memprediksi residual dari Prophet untuk koreksi
    
    Alur:
    - Prophet → baseline prediction (trend + seasonality)
    - LightGBM → menerima prophet_pred sebagai feature untuk koreksi residual
    - Final → prophet_pred + lgbm_residual_pred
    
    Args:
        models (dict): Dictionary berisi 'prophet' dan 'lgbm' models
        start_date (datetime): Tanggal mulai
        end_date (datetime): Tanggal akhir
    
    Returns:
        list: List dictionary dengan hasil prediksi hybrid
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    df_future = pd.DataFrame({'ds': date_range})
    
    # ======== STEP 1: Prepare ALL regressors untuk Prophet ========
    # Get Hijri calendar info
    hijri_info = df_future['ds'].apply(get_hijri_info)
    df_future[['hijri_month', 'hijri_day']] = pd.DataFrame(hijri_info.tolist(), index=df_future.index)
    
    # Regressors
    df_future['is_friday'] = (df_future['ds'].dt.dayofweek == 4).astype(int)
    df_future['is_ramadhan'] = (df_future['hijri_month'] == 9).astype(int)
    df_future['is_last_10_ramadhan'] = ((df_future['hijri_month'] == 9) & (df_future['hijri_day'] >= 21)).astype(int)
    df_future['is_syawal'] = (df_future['hijri_month'] == 10).astype(int)
    df_future['is_zulhijjah'] = (df_future['hijri_month'] == 12).astype(int)
    
    # is_salary_period: Periode gajian (awal: 1-5, tengah: 15-17, akhir: 25-30)
    day_of_month = df_future['ds'].dt.day
    df_future['is_salary_period'] = (
        ((day_of_month >= 1) & (day_of_month <= 5)) |  # Awal bulan
        ((day_of_month >= 15) & (day_of_month <= 17)) |  # Tengah bulan
        ((day_of_month >= 25) & (day_of_month <= 31))    # Akhir bulan
    ).astype(int)
    
    # ======== STEP 2: Prophet Prediction ========
    prophet_model = models['prophet']
    prophet_forecast = prophet_model.predict(df_future)
    prophet_pred = prophet_forecast['yhat'].values
    
    # ======== STEP 3: Prepare features untuk LightGBM ========
    # Time-based features
    df_future['day_of_week'] = df_future['ds'].dt.dayofweek
    df_future['day_of_month'] = df_future['ds'].dt.day
    df_future['month'] = df_future['ds'].dt.month
    df_future['dayofyear'] = df_future['ds'].dt.dayofyear
    df_future['quarter'] = df_future['ds'].dt.quarter
    df_future['prophet_pred'] = prophet_pred
    
    # ======== STEP 4: LightGBM Prediction (Residual Correction) ========
    lgbm_model = models['lgbm']
    # Features harus sama dengan saat training
    lgbm_features = [
        'day_of_week', 'day_of_month', 'month', 'dayofyear', 'quarter',
        'hijri_month', 
        'is_friday', 'is_ramadhan', 'is_last_10_ramadhan', 'is_syawal', 'is_zulhijjah',
        'is_salary_period',
        'prophet_pred'
    ]
    lgbm_residual_pred = lgbm_model.predict(df_future[lgbm_features])
    
    # ======== STEP 5: Ensemble - Prophet baseline + LightGBM residual correction ========
    # LightGBM memprediksi residual, jadi:
    # final_pred = prophet_pred + lgbm_residual_pred (additive)
    ensemble_pred = prophet_pred + lgbm_residual_pred
    
    # Store predictions
    df_future['prophet_prediction'] = prophet_pred
    df_future['lgbm_residual'] = lgbm_residual_pred
    df_future['ensemble_prediction'] = ensemble_pred
    df_future['predicted_donation'] = ensemble_pred
    
    # Ensure non-negative predictions
    df_future['predicted_donation'] = df_future['predicted_donation'].apply(
        lambda x: max(0, round(float(x), 2))
    )
    df_future['prophet_prediction'] = df_future['prophet_prediction'].apply(
        lambda x: max(0, round(float(x), 2))
    )
    df_future['lgbm_residual'] = df_future['lgbm_residual'].apply(
        lambda x: round(float(x), 2)
    )
    
    df_future['ds'] = df_future['ds'].dt.strftime('%Y-%m-%d')
    
    # Return hasil prediksi akhir (tanggal + prediksi gabungan)
    result = df_future[['ds', 'predicted_donation']].rename(
        columns={'ds': 'date'}
    ).to_dict(orient='records')
    
    return result


def make_predictions_legacy(model, start_date, end_date):
    """
    Legacy function - untuk kompatibilitas dengan Prophet model saja
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    df_future = pd.DataFrame({'ds': date_range})
    
    predictions = model.predict(df_future)
    
    df_future['predicted_donation'] = predictions['yhat'].values
    df_future['predicted_donation'] = df_future['predicted_donation'].apply(
        lambda x: max(0, round(float(x), 2))
    )
    df_future['ds'] = df_future['ds'].dt.strftime('%Y-%m-%d')
    
    result = df_future.rename(columns={'ds': 'date'}).to_dict(orient='records')
    
    return result
