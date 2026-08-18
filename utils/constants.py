PANDEMI_AWAL  = '2020-03-01'
PANDEMI_AKHIR = '2021-06-01'

WINDFALL_KW = r'hibah|subsidi|warisan|kosmiaty|rek tpq|aswir idris mtr raya'
NONDONASI_KW = r'ansuran|angsuran|pinjaman|cicilan'
DONASI_KW = (r'infak kotak surau|donatur|donaut|dinatur|'
             r'fitrah|qurban|kurban|maulid|isra|mi.?raj|muharram|idul')
RUTIN_KW = (r'honor|gaji|garin|ustad|imam|baca|ayat|sajad|wirid|khatib|muazin|bilal|'
            r'guru|tpq|mengaji|listrik|sampah|kebersihan|\bair\b|mineral|minum|pdam|'
            r'token|sajuak|sajua|\bdus\b|beras|\bkue\b|makan|konsumsi|\bnasi\b|'
            r'infak|undangan|iuran|sumbangan|administrasi|atk|kertas|tinta|fotokopi')
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
