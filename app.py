from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import warnings
import re
from prophet import Prophet
from lightgbm import LGBMRegressor
from hijridate import Gregorian
import joblib
import os

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ---------------- KONFIGURASI ----------------
# kata kunci transaksi keagamaan yg WAJAR bernilai besar -> TIDAK dihapus saat IQR
KEEP_RELIGIUS = r'idul fitri|idul adha|isra|mi.?raj|muharram|maulid|wakaf|qurban|kurban'
# kata kunci pemasukan one-off (windfall) yg TIDAK rutin -> dipisah dari target peramalan
WINDFALL_KW = r'hibah|subsidi|warisan|kosmiaty|rek tpq|aswir idris mtr raya'
# transaksi NON-DONASI (mis. cicilan pinjaman) -> dikeluarkan dari target.
NONDONASI_KW = r'ansuran|angsuran|pinjaman|cicilan'
# kata kunci DONASI rutin yg WAJAR bernilai besar (akumulasi setoran) -> DILINDUNGI
DONASI_KW = (r'infak kotak surau|donatur|donaut|dinatur|'
             r'fitrah|qurban|kurban|maulid|isra|mi.?raj|muharram|idul')

# Uang Keluar
TRANSFER_KW = (r'pindah buku|pinda buku|pindahbuku|pindah|pinjam|pinjaman|kostizam|'
               r'deposito|setor ke kas|transfer ke')
PROYEK_KW = (r'tukang|semen|besi|paku|pasir|granit|keramik|batu bata|\bbata\b|batu kali|'
             r'kerikil|\bcat\b|kuas|pembangunan|bangun|bagun|renovasi|perbaik|rehab|'
             r'\batap\b|seng|loteng|kusen|plafon|baja ringan|coran|\bcor\b|pondasi|'
             r'kubah|menara|mihrab|kayu|triplek|\bpipa\b|kran|kloset|kabel|saklar|'
             r'engsel|gerinda|\bbor\b|\blas\b|gypsum|\baci\b|dempul|esensa|tevere|'
             r'terali|teralis|\bkaca\b|kaligrafi|caligrafi|gubah|tulisan timbul|cctv|'
             r'\btoa\b|pengeras|speker|speaker|sound|naikan daya|\bbox\b|spanduk|'
             r'\bplang\b|tanah kubur|wuduk|wudhu|\bpintu\b|lemari|karpet|jam digital|'
             r'pemanas|kipas|dispenser|tenda|keranda|bangku|koster|\blampu\b|'
             r'bola listrik|\bmesin\b|bahan bangun|bahan bagun|bahan loteng')
RUTIN_KW = (r'honor|gaji|garin|ustad|imam|baca|ayat|sajad|wirid|khatib|muazin|bilal|'
            r'guru|tpq|mengaji|listrik|sampah|kebersihan|\bair\b|mineral|minum|pdam|'
            r'token|sajuak|sajua|\bdus\b|beras|\bkue\b|makan|konsumsi|\bnasi\b|'
            r'infak|undangan|iuran|sumbangan|administrasi|atk|kertas|tinta|fotokopi')


PANDEMI_AWAL  = '2020-03-01'
PANDEMI_AKHIR = '2021-06-01'
REG = ['is_ramadhan', 'is_idulfitri', 'is_iduladha', 'is_event_kecil', 'is_pandemi', 'ramadhan_days']
FEATS = ['lag_1', 'lag_2', 'lag_3', 'roll_mean_3', 'periode'] + REG

LGB_PARAMS = dict(objective='regression_l1', n_estimators=40, learning_rate=0.05,
                  max_depth=2, num_leaves=4, min_child_samples=4,
                  reg_alpha=1.0, reg_lambda=1.0, subsample=0.8,
                  colsample_bytree=0.8, random_state=42, verbose=-1)

_BLN = r'\b(januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember|jan|feb|mar|apr|jun|jul|agu|agt|agus|sep|okt|nov|nof|des)\b'
_NORM = {'januari':1,'jan':1,'februari':2,'feb':2,'maret':3,'mar':3,'april':4,'apr':4,
         'mei':5,'juni':6,'jun':6,'juli':7,'jul':7,'agustus':8,'agu':8,'agt':8,'agus':8,
         'september':9,'sep':9,'oktober':10,'okt':10,'november':11,'nov':11,'nof':11,
         'desember':12,'des':12}

def _bulan_transaksi(teks):
    t = str(teks).lower()
    toks = [_NORM[w] for w in re.findall(_BLN, t)]
    uniq = sorted(set(toks))
    ada_rentang = bool(re.search(r'[-\u2013\u2014]|s\.?/?d|sampai|hingga', t))
    if ada_rentang and len(uniq) == 2:
        a, b = toks[0], toks[-1]
        urut = list(range(a, b + 1)) if a <= b else list(range(a, 13)) + list(range(1, b + 1))
        return sorted(set(urut))
    return uniq

def add_hijri_flags(df):
    df['is_ramadhan'] = 0
    df['is_idulfitri'] = 0
    df['is_iduladha'] = 0
    df['is_event_kecil'] = 0
    df['is_pandemi'] = 0
    df['ramadhan_days'] = 0.0

    for idx, row in df.iterrows():
        g_date = row['TANGGAL']
        if pd.isna(g_date): continue

        try:
            h_date = Gregorian.fromdate(g_date.date()).to_hijri()
            
            if h_date.month == 9:
                df.at[idx, 'is_ramadhan'] = 1
                df.at[idx, 'ramadhan_days'] = 1.0 
            
            if h_date.month == 10 and h_date.day <= 5:
                df.at[idx, 'is_idulfitri'] = 1
                
            if h_date.month == 12:
                df.at[idx, 'is_iduladha'] = 1
                
            if h_date.month in [1, 3, 7]:
                df.at[idx, 'is_event_kecil'] = 1
                
            if PANDEMI_AWAL <= g_date.strftime('%Y-%m-%d') <= PANDEMI_AKHIR:
                df.at[idx, 'is_pandemi'] = 1
                
        except Exception:
            pass
            
    return df

def create_features(df, target_col):
    df['lag_1'] = df[target_col].shift(1).fillna(method='bfill')
    df['lag_2'] = df[target_col].shift(2).fillna(method='bfill')
    df['lag_3'] = df[target_col].shift(3).fillna(method='bfill')
    df['roll_mean_3'] = df[target_col].rolling(window=3).mean().fillna(method='bfill')
    df['periode'] = df['TANGGAL'].dt.month
    return df

def process_income(data):
    df = pd.DataFrame(data)
    df['TANGGAL'] = pd.to_datetime(df['tanggal']).dt.tz_localize(None)
    df['URAIAN'] = df['uraian'].fillna('').astype(str)
    df['DEBET'] = pd.to_numeric(df['nominal'], errors='coerce').fillna(0).astype(float)
    df = df[df['TANGGAL'] >= '2020-01-01'].copy()
    
    Q1 = df['DEBET'].quantile(0.25)
    Q3 = df['DEBET'].quantile(0.75)
    IQR = Q3 - Q1
    batas_atas = Q3 + 1.5 * IQR

    is_donasi = df['URAIAN'].str.lower().str.contains(DONASI_KW, regex=True)
    di_atas_batas = df['DEBET'] > batas_atas
    dihapus_outlier = di_atas_batas & (~is_donasi)
    df_clean = df[~dihapus_outlier].copy()
    
    u = df_clean['URAIAN'].str.lower()
    is_windfall = u.str.contains(WINDFALL_KW, regex=True)
    is_nondonasi = u.str.contains(NONDONASI_KW, regex=True)
    df_clean = df_clean[~(is_windfall | is_nondonasi)].copy()
    
    is_kotak = df_clean['URAIAN'].str.lower().str.contains(r'kotak|celengan', regex=True)
    kotak_df = df_clean[is_kotak].sort_values('TANGGAL').copy()
    df_rest = df_clean[~is_kotak].copy()
    
    baris_baru = []
    for _, r in df_rest.iterrows():
        bln = _bulan_transaksi(r['URAIAN'])
        if len(bln) >= 2:
            ry, rm = r['TANGGAL'].year, r['TANGGAL'].month
            bagi = r['DEBET'] / len(bln)
            for m in bln:
                y = ry if m <= rm else ry - 1
                baris_baru.append({'TANGGAL': pd.Timestamp(y, m, 1), 'DEBET': bagi})
        else:
            baris_baru.append({'TANGGAL': r['TANGGAL'], 'DEBET': r['DEBET']})
            
    prev = pd.Timestamp('2020-01-01')
    for _, r in kotak_df.iterrows():
        cur = r['TANGGAL'].replace(day=1)
        bulan_akum = pd.date_range(prev, cur, freq='MS')
        if len(bulan_akum) == 0:
            bulan_akum = pd.DatetimeIndex([cur])
        bagi_k = r['DEBET'] / len(bulan_akum)
        for mm in bulan_akum:
            baris_baru.append({'TANGGAL': mm, 'DEBET': bagi_k})
        prev = cur + pd.offsets.MonthBegin(1)
        
    df_clean = pd.DataFrame(baris_baru).sort_values('TANGGAL').reset_index(drop=True)
    df_clean = df_clean[df_clean['TANGGAL'] >= '2020-01-01'].reset_index(drop=True)
    
    df_fmt = df_clean.resample('MS', on='TANGGAL').sum().reset_index()
    df_fmt = add_hijri_flags(df_fmt)
    df_fmt = create_features(df_fmt, 'DEBET')
    
    return df_fmt

def process_expense(data):
    df = pd.DataFrame(data)
    df['TANGGAL'] = pd.to_datetime(df['tanggal']).dt.tz_localize(None)
    df['URAIAN'] = df['uraian'].fillna('').astype(str)
    df['KREDIT'] = pd.to_numeric(df['nominal'], errors='coerce').fillna(0).astype(float)
    df = df[df['TANGGAL'] >= '2020-01-01'].copy()
    
    Q1 = df['KREDIT'].quantile(0.25)
    Q3 = df['KREDIT'].quantile(0.75)
    IQR = Q3 - Q1
    batas_atas = Q3 + 1.5 * IQR

    is_rutin = df['URAIAN'].str.lower().str.contains(RUTIN_KW, regex=True)
    di_atas_batas = df['KREDIT'] > batas_atas
    dihapus_outlier = di_atas_batas & (~is_rutin)
    df_clean = df[~dihapus_outlier].copy()
    
    u = df_clean['URAIAN'].str.lower()
    is_transfer = u.str.contains(TRANSFER_KW, regex=True)
    is_proyek = u.str.contains(PROYEK_KW, regex=True)
    pisah_mask = is_transfer | is_proyek
    df_clean = df_clean[~pisah_mask].copy()
    
    baris_baru = []
    for _, r in df_clean.iterrows():
        bln = _bulan_transaksi(r['URAIAN'])
        if len(bln) >= 2:
            ry, rm = r['TANGGAL'].year, r['TANGGAL'].month
            bagi = r['KREDIT'] / len(bln)
            for m in bln:
                y = ry if m <= rm else ry - 1
                baris_baru.append({'TANGGAL': pd.Timestamp(y, m, 1), 'KREDIT': bagi})
        else:
            baris_baru.append({'TANGGAL': r['TANGGAL'], 'KREDIT': r['KREDIT']})
            
    df_clean = pd.DataFrame(baris_baru).sort_values('TANGGAL').reset_index(drop=True)
    df_clean = df_clean[df_clean['TANGGAL'] >= '2020-01-01'].reset_index(drop=True)
    
    df_fmt = df_clean.resample('MS', on='TANGGAL').sum().reset_index()
    df_fmt = add_hijri_flags(df_fmt)
    df_fmt = create_features(df_fmt, 'KREDIT')
    
    return df_fmt

@app.route('/predict/income', methods=['POST'])
def predict_income():
    try:
        data = request.json.get('transactions', [])
        if not data:
            return jsonify({'success': False, 'message': 'No transaction data provided'})
            
        df = process_income(data)
        
        # Modeling on the fly
        p_df = df.rename(columns={'TANGGAL': 'ds', 'DEBET': 'y'})
        prophet = Prophet()
        for r in REG:
            prophet.add_regressor(r)
        prophet.fit(p_df)
        
        # Residual LightGBM
        p_pred = prophet.predict(p_df)
        res_train = p_df['y'] - p_pred['yhat']
        
        lgbm = LGBMRegressor(**LGB_PARAMS)
        lgbm.fit(p_df[FEATS], res_train)
        
        # Predict next 1 month
        last_date = p_df['ds'].max()
        future_ds = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=1, freq='MS')
        fut_df = pd.DataFrame({'ds': future_ds, 'TANGGAL': future_ds})
        fut_df = add_hijri_flags(fut_df)
        
        # Merge recent data for lags
        fut_df['lag_1'] = p_df['y'].iloc[-1]
        fut_df['lag_2'] = p_df['y'].iloc[-2] if len(p_df) > 1 else p_df['y'].iloc[-1]
        fut_df['lag_3'] = p_df['y'].iloc[-3] if len(p_df) > 2 else p_df['y'].iloc[-1]
        fut_df['roll_mean_3'] = p_df['y'].iloc[-3:].mean() if len(p_df) > 2 else p_df['y'].iloc[-1]
        fut_df['periode'] = fut_df['ds'].dt.month
        
        fut_prophet = prophet.predict(fut_df)
        fut_res = lgbm.predict(fut_df[FEATS])
        
        fut_df['Prediksi_Prophet'] = fut_prophet['yhat'].values
        fut_df['Prediksi_Hybrid'] = fut_df['Prediksi_Prophet'] + fut_res
        
        result = []
        for _, row in fut_df.iterrows():
            islam_event = "-"
            if row['is_ramadhan']: islam_event = "Ramadhan"
            elif row['is_idulfitri']: islam_event = "Idul Fitri"
            elif row['is_iduladha']: islam_event = "Idul Adha"
            elif row['is_event_kecil']: 
                try:
                    h_m = Gregorian.fromdate(row['ds'].date()).to_hijri().month
                    if h_m == 1: islam_event = "Tahun Baru Islam (Muharram)"
                    elif h_m == 3: islam_event = "Maulid Nabi (Rabi'ul Awal)"
                    elif h_m == 7: islam_event = "Isra' Mi'raj (Rajab)"
                    else: islam_event = "Event Kecil"
                except:
                    islam_event = "Event Kecil"
            
            result.append({
                'Tanggal': row['ds'].strftime('%Y-%m-%d'),
                'Hari_Besar_Islam': islam_event,
                'Prediksi_Prophet': float(row['Prediksi_Prophet']),
                'Prediksi_Hybrid': float(row['Prediksi_Hybrid'])
            })
            
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/predict/expense', methods=['POST'])
def predict_expense():
    try:
        data = request.json.get('transactions', [])
        if not data:
            return jsonify({'success': False, 'message': 'No transaction data provided'})
            
        df = process_expense(data)
        
        # Modeling on the fly
        p_df = df.rename(columns={'TANGGAL': 'ds', 'KREDIT': 'y'})
        prophet = Prophet()
        for r in REG:
            prophet.add_regressor(r)
        prophet.fit(p_df)
        
        # Residual LightGBM
        p_pred = prophet.predict(p_df)
        res_train = p_df['y'] - p_pred['yhat']
        
        lgbm = LGBMRegressor(**LGB_PARAMS)
        lgbm.fit(p_df[FEATS], res_train)
        
        # Predict next 1 month
        last_date = p_df['ds'].max()
        future_ds = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=1, freq='MS')
        fut_df = pd.DataFrame({'ds': future_ds, 'TANGGAL': future_ds})
        fut_df = add_hijri_flags(fut_df)
        
        # Merge recent data for lags
        fut_df['lag_1'] = p_df['y'].iloc[-1]
        fut_df['lag_2'] = p_df['y'].iloc[-2] if len(p_df) > 1 else p_df['y'].iloc[-1]
        fut_df['lag_3'] = p_df['y'].iloc[-3] if len(p_df) > 2 else p_df['y'].iloc[-1]
        fut_df['roll_mean_3'] = p_df['y'].iloc[-3:].mean() if len(p_df) > 2 else p_df['y'].iloc[-1]
        fut_df['periode'] = fut_df['ds'].dt.month
        
        fut_prophet = prophet.predict(fut_df)
        fut_res = lgbm.predict(fut_df[FEATS])
        
        fut_df['Prediksi_Prophet'] = fut_prophet['yhat'].values
        fut_df['Prediksi_Hybrid'] = fut_df['Prediksi_Prophet'] + fut_res
        
        result = []
        for _, row in fut_df.iterrows():
            islam_event = "-"
            if row['is_ramadhan']: islam_event = "Ramadhan"
            elif row['is_idulfitri']: islam_event = "Idul Fitri"
            elif row['is_iduladha']: islam_event = "Idul Adha"
            elif row['is_event_kecil']: 
                try:
                    h_m = Gregorian.fromdate(row['ds'].date()).to_hijri().month
                    if h_m == 1: islam_event = "Tahun Baru Islam (Muharram)"
                    elif h_m == 3: islam_event = "Maulid Nabi (Rabi'ul Awal)"
                    elif h_m == 7: islam_event = "Isra' Mi'raj (Rajab)"
                    else: islam_event = "Event Kecil"
                except:
                    islam_event = "Event Kecil"
            
            result.append({
                'Tanggal': row['ds'].strftime('%Y-%m-%d'),
                'Hari_Besar_Islam': islam_event,
                'Prediksi_Prophet': float(row['Prediksi_Prophet']),
                'Prediksi_Hybrid': float(row['Prediksi_Hybrid'])
            })
            
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
