import pandas as pd

def prepare_dataframe(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Limpia y genera todas las columnas de etiqueta necesarias.
    """

    df = df.copy()
    df['lyrics']       = df[cfg.columns.LYRICS_COL]
    df['lyrics']       = df['lyrics'].astype(str).str.strip()
    df['lyrics_clean'] = df['lyrics'].str.lower().str.split().str.join(' ')

    if cfg.flags.USE_GLOSS and cfg.columns.GLOSS_COL in df.columns:
        df['gloss_clean'] = (df[cfg.columns.GLOSS_COL]
                             .fillna('')
                             .astype(str)
                             .str.lower()
                             .str.split()
                             .str.join(' '))
        print("[FLAG]: Using Glosses for Training")
    else:
        df['gloss_clean'] = ''   # void column -> comportamiento idéntico al anterior
        print("[FLAG]: Not using Glosses for Training")

    
    # ── Task 1: binary misoginia ──────────────────────────────────────────
    label_map_t1 = {'M': 1, 'NM': 0, True: 1, False: 0, 1: 1, 0: 0,
                    'misogynistic': 1, 'non-misogynistic': 0}
    if 'is_misogynistic' in df.columns:
        df['label_t1'] = df['is_misogynistic'].map(label_map_t1).fillna(0).astype(int)
    else:
        df['label_t1'] = 0

    # ── Task 2: tipo multilabel ────────────────────────────────────────────
    for col in ['type_hate', 'type_violence', 'type_sexualization']:
        df[col] = df[col].fillna(0).astype(int) if col in df.columns else 0

    df['nothing'] = ((df['type_hate'] == 0) &
                     (df['type_violence'] == 0) &
                     (df['type_sexualization'] == 0)).astype(int)

    # ── Task 3: estereotipo binario ────────────────────────────────────────
    label_map_t3 = {'Y': 1, 'N': 0, True: 1, False: 0, 1: 1, 0: 0}
    if 'has_gender_stereotype' in df.columns:
        df['label_t3'] = df['has_gender_stereotype'].map(label_map_t3).fillna(0).astype(int)
    else:
        df['label_t3'] = 0

    return df.dropna(subset=['lyrics_clean'])
"""




import pandas as pd

def deduplicate_lyrics(text: str) -> str:
    """
    Elimina párrafos duplicados globales y versos idénticos consecutivos.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    paragraphs = text.split('\n\n')
    unique_paragraphs = []
    seen_paragraphs = set()

    for p in paragraphs:
        p_strip = p.strip()
        if not p_strip or p_strip in seen_paragraphs:
            continue
            
        lines = p_strip.split('\n')
        clean_lines = []
        for i, line in enumerate(lines):
            l_strip = line.strip()
            if not l_strip:
                continue
            if i == 0 or l_strip != lines[i-1].strip():
                clean_lines.append(l_strip)
        
        unique_paragraphs.append("\n".join(clean_lines))
        seen_paragraphs.add(p_strip)

    return "\n\n".join(unique_paragraphs)

def prepare_dataframe(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Limpia, deduplica y genera métricas de longitud de texto.
    """
    df = df.copy()
    
    # 1. Preparación inicial y métricas PRE-limpieza
    df['lyrics'] = df[cfg.columns.LYRICS_COL].astype(str).str.strip()
    words_pre = df['lyrics'].str.split().str.len()
    avg_pre = words_pre.mean()

    # 2. Ejecución de limpieza y deduplicación
    df['lyrics'] = df['lyrics'].apply(deduplicate_lyrics)
    df['lyrics_clean'] = df['lyrics'].str.lower().str.split().str.join(' ')

    # 3. Métricas POST-limpieza
    words_post = df['lyrics_clean'].str.split().str.len()
    avg_post = words_post.mean()

    print(f"[METRICS]: Average words per song (PRE):  {avg_pre:.2f}")
    print(f"[METRICS]: Average words per song (POST): {avg_post:.2f}")
    print(f"[METRICS]: Token reduction: {((avg_pre - avg_post) / avg_pre) * 100:.2f}%")

    # ── Manejo de Glosses ──────────────────────────────────────────────────
    if cfg.flags.USE_GLOSS and cfg.columns.GLOSS_COL in df.columns:
        df['gloss_clean'] = (df[cfg.columns.GLOSS_COL]
                             .fillna('')
                             .astype(str)
                             .str.lower()
                             .str.split()
                             .str.join(' '))
        print("[FLAG]: Using Glosses for Training")
    else:
        df['gloss_clean'] = ''
        print("[FLAG]: Not using Glosses for Training")

    # ── Task 1: binary misoginia ──────────────────────────────────────────
    label_map_t1 = {'M': 1, 'NM': 0, True: 1, False: 0, 1: 1, 0: 0,
                    'misogynistic': 1, 'non-misogynistic': 0}
    if 'is_misogynistic' in df.columns:
        df['label_t1'] = df['is_misogynistic'].map(label_map_t1).fillna(0).astype(int)
    else:
        df['label_t1'] = 0

    # ── Task 2: tipo multilabel ────────────────────────────────────────────
    for col in ['type_hate', 'type_violence', 'type_sexualization']:
        df[col] = df[col].fillna(0).astype(int) if col in df.columns else 0

    df['nothing'] = ((df['type_hate'] == 0) &
                     (df['type_violence'] == 0) &
                     (df['type_sexualization'] == 0)).astype(int)

    # ── Task 3: estereotipo binario ────────────────────────────────────────
    label_map_t3 = {'Y': 1, 'N': 0, True: 1, False: 0, 1: 1, 0: 0}
    if 'has_gender_stereotype' in df.columns:
        df['label_t3'] = df['has_gender_stereotype'].map(label_map_t3).fillna(0).astype(int)
    else:
        df['label_t3'] = 0

    return df.dropna(subset=['lyrics_clean'])