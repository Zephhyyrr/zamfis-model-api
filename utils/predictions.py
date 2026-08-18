import pandas as pd
import numpy as np

from .preprocessing import preprocess_transactions
from .hijri_utils import add_hijri_flags, get_event_kecil_name


def prophet_predict(m, frame, reg):
    d = frame[['ds']].copy()
    for r in reg:
        d[r] = frame[r].values
    return m.predict(d)['yhat'].values


def process_transaction_history(transactions, prophet, REG, type_name):
    if not transactions:
        return None
        
    df = pd.DataFrame(transactions)
    if df.empty:
        return None
        
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    df['tanggal'] = df['tanggal'].apply(lambda x: x.tz_localize(None) if x.tzinfo else x)
    df = df.rename(columns={'tanggal': 'ds', 'nominal': 'y'})
    
    # Terapkan Preprocessing (IQR, dll)
    df = preprocess_transactions(df, type_name)
    if df.empty:
        return None
    
    # Resample per bulan
    df_monthly = df.set_index('ds').resample('MS').sum().reset_index()
    
    # Selalu hapus bulan terakhir di data (dianggap sebagai bulan berjalan yang belum selesai)
    # agar tidak mempengaruhi fitur lag
    if not df_monthly.empty:
        max_date = df_monthly['ds'].max()
        print(f"Peringatan: Mengeluarkan bulan terakhir di data ({max_date.strftime('%Y-%m')}) dari history karena dianggap belum selesai.")
        df_monthly = df_monthly[df_monthly['ds'] < max_date]
            
    if df_monthly.empty:
        return None
    
    # Tambahkan flag kalender hijriah
    df_monthly = add_hijri_flags(df_monthly)
    
    # Dapatkan prediksi baseline prophet untuk semua data history
    df_monthly['prophet_pred'] = prophet_predict(prophet, df_monthly, REG)
    
    # Hitung residual nyata (Aktual - Prophet)
    df_monthly['resid'] = df_monthly['y'] - df_monthly['prophet_pred']
    
    return df_monthly


def make_predictions(M, type_name, months_ahead=1, transactions=None):
    prophet = M['prophet']
    lgbm    = M['lgbm']
    a       = M['alpha']
    REG     = M['reg']
    FEATS   = M['feats']

    # Tentukan bulan terakhir dari raw transaksi (sebelum di-drop)
    original_max_date = None
    if transactions:
        df_temp = pd.DataFrame(transactions)
        if not df_temp.empty:
            df_temp['tanggal'] = pd.to_datetime(df_temp['tanggal'])
            df_temp['tanggal'] = df_temp['tanggal'].apply(lambda x: x.tz_localize(None) if x.tzinfo else x)
            original_max_date = df_temp['tanggal'].max().replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    # Jika tidak ada, gunakan M['history'] default bawaan .pkl
    hist = process_transaction_history(transactions, prophet, REG, type_name)
    if hist is None or hist.empty:
        hist = M['history'].copy()
        
    last = hist['ds'].max()

    if original_max_date is not None and original_max_date > last:
        target_start = original_max_date + pd.offsets.MonthBegin(1)
    else:
        target_start = last + pd.offsets.MonthBegin(1)

    target_end = target_start + pd.offsets.MonthBegin(months_ahead - 1)
    
    # Generate bulan-bulan yang akan diprediksi (mulai 1 bulan setelah 'last', sampai 'target_end')
    future_ds = pd.date_range(last + pd.offsets.MonthBegin(1), target_end, freq='MS')
    fut = pd.DataFrame({'ds': future_ds})
    fut = add_hijri_flags(fut)

    # Prediksi baseline dari Prophet
    fut['prophet_pred'] = prophet_predict(prophet, fut, REG)

    # Siapkan resid_hist dari history pkl (persis seperti notebook)
    resid_hist = list(hist['resid'].values)
    
    data_quality_warning = None
    if len(resid_hist) < 3:
        data_quality_warning = "History residual kurang dari 3 bulan, akurasi forecast berkurang"

    hasil = []
    prophet_hasil = []

    for _, row in fut.iterrows():
        fitur = {
            'lag_1':       resid_hist[-1] if len(resid_hist) >= 1 else 0,
            'lag_2':       resid_hist[-2] if len(resid_hist) >= 2 else 0,
            'lag_3':       resid_hist[-3] if len(resid_hist) >= 3 else 0,
            'roll_mean_3': np.mean(resid_hist[-3:]) if len(resid_hist) >= 3 else 0,
            'periode':     row['ds'].month
        }
        for r in REG:
            fitur[r] = row[r]

        x  = pd.DataFrame([fitur])[FEATS]
        rh = lgbm.predict(x)[0]
        resid_hist.append(rh)

        p_val      = row['prophet_pred']
        hybrid_val = max(0.0, p_val + a * rh)

        prophet_hasil.append(max(0.0, p_val))
        hasil.append(hybrid_val)

    fut['prophet_prediction'] = prophet_hasil
    fut['predicted_donation'] = hasil
    if data_quality_warning:
        fut['data_quality_warning'] = data_quality_warning
    
    LABEL_MAP = {
        'is_ramadhan':    'Ramadhan',
        'is_idulfitri':   'Idul Fitri',
        'is_iduladha':    'Idul Adha',
        'is_event_kecil': None,   # diisi dinamis oleh get_event_kecil_name
        'is_pandemi':     'Pandemi',
        'ramadhan_days':  'Bulan Ramadhan',
    }

    def get_hijri_label(row):
        tags = []
        for col in REG:
            val = row.get(col, 0)
            if not val:
                continue
            if col == 'is_event_kecil':
                # Tentukan nama event kecil yang spesifik
                tags.append(get_event_kecil_name(row['ds']))
            elif col == 'ramadhan_days':
                pass  # sudah diwakili is_ramadhan, skip agar tidak dobel
            else:
                label = LABEL_MAP.get(col, col.replace('is_', ''))
                tags.append(label)
        return ', '.join(tags) if tags else '-'

    fut['hijri_events'] = fut.apply(get_hijri_label, axis=1)
    fut['ds'] = fut['ds'].dt.strftime('%Y-%m-%d')

    cols_to_export = ['ds', 'prophet_prediction', 'predicted_donation', 'hijri_events']
    if data_quality_warning:
        cols_to_export.append('data_quality_warning')
        
    result = fut[cols_to_export].rename(
        columns={'ds': 'date'}
    ).to_dict(orient='records')
    
    # Filter hanya kembalikan hasil mulai dari target_start
    target_start_str = target_start.strftime('%Y-%m-%d')
    result = [r for r in result if r['date'] >= target_start_str]

    return result
