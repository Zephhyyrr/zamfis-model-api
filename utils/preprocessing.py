import pandas as pd
import re
from .constants import WINDFALL_KW, NONDONASI_KW, DONASI_KW, RUTIN_KW, TRANSFER_KW, PROYEK_KW

def preprocess_transactions(df, type_name):
    if df.empty:
        return df

    if 'uraian' not in df.columns:
        if 'keterangan' in df.columns:
            df['uraian'] = df['keterangan']
        else:
            df['uraian'] = ''
            
    df['uraian_lower'] = df['uraian'].fillna('').astype(str).str.lower()
    
    if type_name == 'income':
        # 1. Hapus Outlier IQR (hanya non-donasi)
        Q1 = df['y'].quantile(0.25)
        Q3 = df['y'].quantile(0.75)
        IQR = Q3 - Q1
        batas_bawah = Q1 - 1.5 * IQR
        batas_atas  = Q3 + 1.5 * IQR
        
        is_donasi = df['uraian_lower'].str.contains(DONASI_KW, regex=True)
        di_atas_batas = df['y'] > batas_atas
        dihapus_outlier = di_atas_batas & (~is_donasi)
        
        df_clean = df[~dihapus_outlier].copy()
        
        # 2. Pisahkan windfall & non-donasi
        u = df_clean['uraian_lower']
        is_windfall = u.str.contains(WINDFALL_KW, regex=True)
        is_nondonasi = u.str.contains(NONDONASI_KW, regex=True)
        df_clean = df_clean[~(is_windfall | is_nondonasi)].copy()
        
        # 3. Pisahkan infak kotak
        is_kotak = df_clean['uraian_lower'].str.contains(r'kotak|celengan', regex=True)
        kotak_df = df_clean[is_kotak].sort_values('ds').copy()
        df_rest = df_clean[~is_kotak].copy()
        
        # 4. Redistribusi transaksi bundel
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
            
        baris_baru = []
        for _, r in df_rest.iterrows():
            bln = _bulan_transaksi(r['uraian_lower'])
            if len(bln) >= 2:
                ry, rm = r['ds'].year, r['ds'].month
                bagi = r['y'] / len(bln)
                for m in bln:
                    y_year = ry if m <= rm else ry - 1
                    baris_baru.append({'ds': pd.Timestamp(y_year, m, 1), 'uraian': str(r.get('uraian', '')) + ' [redistribusi]', 'y': bagi})
            else:
                baris_baru.append({'ds': r['ds'], 'uraian': r.get('uraian', ''), 'y': r['y']})
                
        # 5. Haluskan infak kotak
        if not df_clean.empty:
            prev = df_clean['ds'].min().replace(day=1)
            for _, r in kotak_df.iterrows():
                cur = r['ds'].replace(day=1)
                bulan_akum = pd.date_range(prev, cur, freq='MS')
                if len(bulan_akum) == 0:
                    bulan_akum = pd.DatetimeIndex([cur])
                bagi_k = r['y'] / len(bulan_akum)
                for mm in bulan_akum:
                    baris_baru.append({'ds': mm, 'uraian': str(r.get('uraian', '')) + ' [kotak-halus]', 'y': bagi_k})
                prev = cur + pd.offsets.MonthBegin(1)
                
        if baris_baru:
            df_clean = pd.DataFrame(baris_baru).sort_values('ds').reset_index(drop=True)
            return df_clean
        else:
            return df_clean
    elif type_name == 'expense':
        # 1. Hapus Outlier IQR (hanya non-rutin)
        Q1 = df['y'].quantile(0.25)
        Q3 = df['y'].quantile(0.75)
        IQR = Q3 - Q1
        batas_atas  = Q3 + 1.5 * IQR
        
        is_rutin = df['uraian_lower'].str.contains(RUTIN_KW, regex=True)
        di_atas_batas = df['y'] > batas_atas
        dihapus_outlier = di_atas_batas & (~is_rutin)
        
        df_clean = df[~dihapus_outlier].copy()
        
        # 2. Pisahkan Transfer/Pinjaman & Proyek
        u = df_clean['uraian_lower']
        is_transfer = u.str.contains(TRANSFER_KW, regex=True)
        is_proyek   = u.str.contains(PROYEK_KW, regex=True)
        pisah_mask  = is_transfer | is_proyek
        df_clean = df_clean[~pisah_mask].copy()

        # 3. Redistribusi transaksi bundel multi-bulan
        _BLN = r'\b(januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember|jan|feb|mar|apr|jun|jul|agu|agt|agus|sep|okt|nov|nof|des)\b'
        _NORM = {'januari':1,'jan':1,'februari':2,'feb':2,'maret':3,'mar':3,'april':4,'apr':4,
                 'mei':5,'juni':6,'jun':6,'juli':7,'jul':7,'agustus':8,'agu':8,'agt':8,'agus':8,
                 'september':9,'sep':9,'oktober':10,'okt':10,'november':11,'nov':11,'nof':11,
                 'desember':12,'des':12}
                 
        def _bulan_transaksi(teks):
            t = str(teks).lower()
            toks = [_NORM[w] for w in re.findall(_BLN, t)]
            return sorted(set(toks))
            
        baris_baru = []
        for _, r in df_clean.iterrows():
            bln = _bulan_transaksi(r['uraian_lower'])
            if len(bln) >= 2:
                ry, rm = r['ds'].year, r['ds'].month
                bagi = r['y'] / len(bln)
                for m in bln:
                    y_year = ry if m <= rm else ry - 1
                    baris_baru.append({'ds': pd.Timestamp(y_year, m, 1), 'uraian': str(r.get('uraian', '')) + ' [redistribusi]', 'y': bagi})
            else:
                baris_baru.append({'ds': r['ds'], 'uraian': r.get('uraian', ''), 'y': r['y']})
                
        if baris_baru:
            df_clean = pd.DataFrame(baris_baru).sort_values('ds').reset_index(drop=True)
            return df_clean
        else:
            return df_clean

    return df
