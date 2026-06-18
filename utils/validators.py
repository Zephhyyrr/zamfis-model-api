from datetime import datetime

def validate_date_range(start_str, end_str):
    """
    Memvalidasi format tanggal (YYYY-MM-DD) dan memastikan rentang waktu 
    antara tanggal mulai dan tanggal akhir tidak melebihi 365 hari (1 tahun).
    
    Args:
        start_str (str): Tanggal mulai dalam format YYYY-MM-DD
        end_str (str): Tanggal akhir dalam format YYYY-MM-DD
    """
    try:
        start_date = datetime.strptime(start_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_str, '%Y-%m-%d')
        
        if start_date > end_date:
            return False, None, None, "Tanggal mulai tidak boleh lebih besar dari tanggal akhir."
            
        if (end_date - start_date).days > 365:
            return False, None, None, "Rentang tanggal maksimal adalah 1 tahun (365 hari)."
            
        return True, start_date, end_date, None
    except ValueError:
        return False, None, None, "Format tanggal salah. Gunakan YYYY-MM-DD."
