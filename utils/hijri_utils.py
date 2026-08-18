import pandas as pd
from hijridate import Gregorian
from .constants import PANDEMI_AWAL, PANDEMI_AKHIR

def add_hijri_flags(frame):
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


def get_event_kecil_name(ds):
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
