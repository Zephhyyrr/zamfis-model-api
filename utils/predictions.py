import pandas as pd
import numpy as np
from hijridate import Gregorian

PANDEMI_AWAL  = '2020-03-01'
PANDEMI_AKHIR = '2021-06-01'


def add_hijri_flags(frame):
    """
    Menambahkan penanda hari besar Islam (Hijriah) ke dataframe bulanan.
    Persis seperti fungsi di notebook modeling.
    """
    pan_lo = pd.Timestamp(PANDEMI_AWAL)
    pan_hi = pd.Timestamp(PANDEMI_AKHIR)
    ram_l, idf_l, ida_l, kecil_l, pan_l, ramd_l = [], [], [], [], [], []

    for ts in frame['ds']:
        ts_naive = ts.tz_localize(None) if hasattr(ts, 'tzinfo') and ts.tzinfo else ts
        rng = pd.date_range(ts_naive, ts_naive + pd.offsets.MonthEnd(0))
        ram = idf = ida = kecil = 0
        ram_days = 0
        for d in rng:
            try:
                h = Gregorian(d.year, d.month, d.day).to_hijri()
                if h.month == 9:
                    ram = 1; ram_days += 1
                if h.month == 10 and h.day == 1:
                    idf = 1
                if h.month == 12 and h.day == 10:
                    ida = 1
                if h.month == 3 and 10 <= h.day <= 14:
                    kecil = 1
                if h.month == 7 and 25 <= h.day <= 29:
                    kecil = 1
                if h.month == 1 and h.day <= 12:
                    kecil = 1
            except Exception:
                pass
        ram_l.append(ram); idf_l.append(idf); ida_l.append(ida); kecil_l.append(kecil)
        pan_l.append(1 if (pan_lo <= ts_naive <= pan_hi) else 0)
        ramd_l.append(ram_days / 30.0)

    frame['is_ramadhan']    = ram_l
    frame['is_idulfitri']   = idf_l
    frame['is_iduladha']    = ida_l
    frame['is_event_kecil'] = kecil_l
    frame['is_pandemi']     = pan_l
    frame['ramadhan_days']  = ramd_l
    return frame


def prophet_predict(m, frame, reg):
    d = frame[['ds']].copy()
    for r in reg:
        d[r] = frame[r].values
    return m.predict(d)['yhat'].values


def get_event_kecil_name(ds):
    """
    Menentukan nama spesifik event kecil Islam dalam bulan tersebut.
    Sesuai logika add_hijri_flags di notebook:
    - Maulid Nabi  : Rabiul Awal bulan 3, hari 10-14
    - Isra Mi'raj  : Rajab bulan 7, hari 25-29
    - Muharram/Asyura: Muharram bulan 1, hari 1-12
    """
    rng = pd.date_range(ds, ds + pd.offsets.MonthEnd(0))
    names = []
    for d in rng:
        try:
            h = Gregorian(d.year, d.month, d.day).to_hijri()
            if h.month == 3 and 10 <= h.day <= 14 and 'Maulid Nabi' not in names:
                names.append('Maulid Nabi')
            if h.month == 7 and 25 <= h.day <= 29 and "Isra Mi'raj" not in names:
                names.append("Isra Mi'raj")
            if h.month == 1 and h.day <= 12 and 'Muharram/Asyura' not in names:
                names.append('Muharram/Asyura')
        except Exception:
            pass
    return ', '.join(names) if names else 'Event Kecil'


def process_transaction_history(transactions, prophet, REG):
    """
    Mengambil raw transaksi dari database, mengubahnya ke format bulanan,
    dan menghitung residual terhadap prediksi Prophet baseline.
    """
    if not transactions:
        return None
        
    df = pd.DataFrame(transactions)
    if df.empty:
        return None
        
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    df['tanggal'] = df['tanggal'].apply(lambda x: x.tz_localize(None) if x.tzinfo else x)
    df = df.rename(columns={'tanggal': 'ds', 'nominal': 'y'})
    
    # Resample per bulan
    df_monthly = df.set_index('ds').resample('MS').sum().reset_index()
    
    # Tambahkan flag kalender hijriah
    df_monthly = add_hijri_flags(df_monthly)
    
    # Dapatkan prediksi baseline prophet untuk semua data history
    df_monthly['prophet_pred'] = prophet_predict(prophet, df_monthly, REG)
    
    # Hitung residual nyata (Aktual - Prophet)
    df_monthly['resid'] = df_monthly['y'] - df_monthly['prophet_pred']
    
    return df_monthly


def make_predictions(M, months_ahead=1, transactions=None):
    """
    Melakukan prediksi hybrid PERSIS seperti fungsi forecast_future di notebook.

    Menggunakan M['history'] dari dalam file .pkl sebagai acuan lag/residual.
    TIDAK menggunakan data dari database.
    """
    prophet = M['prophet']
    lgbm    = M['lgbm']
    a       = M['alpha']
    REG     = M['reg']
    FEATS   = M['feats']

    # Jika ada transaksi dari database, gunakan itu untuk mendapatkan history & residual terbaru
    # Jika tidak ada, gunakan M['history'] default bawaan .pkl
    hist = process_transaction_history(transactions, prophet, REG)
    if hist is None or hist.empty:
        hist = M['history'].copy()
        
    last = hist['ds'].max()

    # Generate bulan-bulan yang akan diprediksi (mulai 1 bulan setelah data terakhir)
    future_ds = pd.date_range(last + pd.offsets.MonthBegin(1), periods=months_ahead, freq='MS')
    fut = pd.DataFrame({'ds': future_ds})
    fut = add_hijri_flags(fut)

    # Prediksi baseline dari Prophet
    fut['prophet_pred'] = prophet_predict(prophet, fut, REG)

    # Siapkan resid_hist dari history pkl (persis seperti notebook)
    resid_hist = list(hist['resid'].values)

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
    
    # Buat label kalender Hijriah seperti di notebook Colab:
    # tag = [n.replace('is_', '') for n in REG if r[n]]
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

    result = fut[['ds', 'prophet_prediction', 'predicted_donation', 'hijri_events']].rename(
        columns={'ds': 'date'}
    ).to_dict(orient='records')

    return result
