import re
import pandas as pd
from collections import Counter


# ─── Utilidades de limpieza ────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Limpieza general: tags estructurales, caracteres raros, espacios y encoding.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    # 1. Tags tipo [Verso 1], [Coro], [Chorus], (Intro), etc.
    text = re.sub(r'[\[\(][^\]\)]{1,40}[\]\)]', '', text)

    # 2. Caracteres de control y no imprimibles (salvo \n)
    text = re.sub(r'[^\S\n]+', ' ', text)          # espacios múltiples → uno
    text = re.sub(r'[\x00-\x08\x0b-\x1f\x7f]', '', text)  # control chars

    # 3. Caracteres raros frecuentes en scraping
    text = text.replace('\u200b', '').replace('\u00ad', '')  # zero-width, soft-hyphen
    text = text.replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')

    # 4. Puntuación excesiva: "!!!!!!" → "!", "......" → "..."
    text = re.sub(r'([!?]){2,}', r'\1', text)
    text = re.sub(r'\.{4,}', '...', text)

    # 5. Líneas vacías múltiples → separador de párrafo limpio
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def get_ngrams(text: str, n: int) -> list[tuple]:
    words = text.lower().split()
    return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]


def has_repeated_ngram(paragraph: str, global_ngrams: Counter, n: int = 4, threshold: int = 2) -> bool:
    """
    Devuelve True si el párrafo contiene un n-grama que ya aparece
    `threshold` o más veces en el corpus global acumulado.
    """
    for ngram in get_ngrams(paragraph, n):
        if global_ngrams[ngram] >= threshold:
            return True
    return False


def deduplicate_lyrics(text: str, ngram_n: int = 4, ngram_threshold: int = 2) -> str:
    """
    Elimina:
    - Párrafos duplicados exactos
    - Versos consecutivos idénticos
    - Párrafos con n-gramas muy repetidos respecto al resto de la canción
    """
    if not isinstance(text, str) or not text.strip():
        return text

    paragraphs = text.split('\n\n')

    # ── Paso 1: limpiar líneas vacías y versos consecutivos idénticos ─────────
    cleaned_paragraphs = []
    for p in paragraphs:
        lines = [l.strip() for l in p.strip().split('\n') if l.strip()]
        deduped_lines = [lines[0]] if lines else []
        for line in lines[1:]:
            if line.lower() != deduped_lines[-1].lower():
                deduped_lines.append(line)
        if deduped_lines:
            cleaned_paragraphs.append('\n'.join(deduped_lines))

    # ── Paso 2: eliminar párrafos duplicados exactos ──────────────────────────
    seen_exact = set()
    no_exact_dups = []
    for p in cleaned_paragraphs:
        key = p.lower().strip()
        if key not in seen_exact:
            seen_exact.add(key)
            no_exact_dups.append(p)

    # ── Paso 3: deduplicación por n-gramas ────────────────────────────────────
    # Construimos Counter global con todos los párrafos primero
    global_ngrams: Counter = Counter()
    for p in no_exact_dups:
        global_ngrams.update(get_ngrams(p, ngram_n))

    final_paragraphs = []
    seen_ngrams: Counter = Counter()  # n-gramas ya "usados" en el output final

    for p in no_exact_dups:
        p_ngrams = get_ngrams(p, ngram_n)
        # Si algún n-grama de este párrafo ya aparece en el output final, es repetición
        if any(seen_ngrams[ng] >= ngram_threshold for ng in p_ngrams):
            continue
        seen_ngrams.update(p_ngrams)
        final_paragraphs.append(p)

    return '\n\n'.join(final_paragraphs)


# ─── Pipeline principal ────────────────────────────────────────────────────────

def prepare_dataframe(df: pd.DataFrame, cfg) -> pd.DataFrame:
    df = df.copy()

    # 1. Texto crudo y métricas PRE
    df['lyrics'] = df[cfg.columns.LYRICS_COL].astype(str).str.strip()
    avg_pre = df['lyrics'].str.split().str.len().mean()

    # 2. Limpieza general + deduplicación
    df['lyrics'] = df['lyrics'].apply(clean_text)
    df['lyrics'] = df['lyrics'].apply(deduplicate_lyrics)

    # 3. Versión normalizada para modelos
    df['lyrics_clean'] = df['lyrics'].str.lower().str.split().str.join(' ')

    # 4. Métricas POST
    avg_post = df['lyrics_clean'].str.split().str.len().mean()
    reduction = ((avg_pre - avg_post) / avg_pre) * 100
    print(f"[METRICS] Palabras promedio PRE:  {avg_pre:.2f}")
    print(f"[METRICS] Palabras promedio POST: {avg_post:.2f}")
    print(f"[METRICS] Reducción de tokens:    {reduction:.2f}%")

    # ── Glosses ───────────────────────────────────────────────────────────────
    if cfg.flags.USE_GLOSS and cfg.columns.GLOSS_COL in df.columns:
        df['gloss_clean'] = (df[cfg.columns.GLOSS_COL]
                             .fillna('').astype(str)
                             .str.lower().str.split().str.join(' '))
        print("[FLAG] Usando Glosses para entrenamiento")
    else:
        df['gloss_clean'] = ''
        print("[FLAG] Sin Glosses")

    # ── Task 1: misoginia binaria ─────────────────────────────────────────────
    label_map_t1 = {'M': 1, 'NM': 0, True: 1, False: 0, 1: 1, 0: 0,
                    'misogynistic': 1, 'non-misogynistic': 0}
    df['label_t1'] = (df['is_misogynistic'].map(label_map_t1).fillna(0).astype(int)
                      if 'is_misogynistic' in df.columns else 0)

    # ── Task 2: tipo multilabel ───────────────────────────────────────────────
    for col in ['type_hate', 'type_violence', 'type_sexualization']:
        df[col] = df[col].fillna(0).astype(int) if col in df.columns else 0
    df['nothing'] = ((df['type_hate'] == 0) &
                     (df['type_violence'] == 0) &
                     (df['type_sexualization'] == 0)).astype(int)

    # ── Task 3: estereotipo binario ───────────────────────────────────────────
    label_map_t3 = {'Y': 1, 'N': 0, True: 1, False: 0, 1: 1, 0: 0}
    df['label_t3'] = (df['has_gender_stereotype'].map(label_map_t3).fillna(0).astype(int)
                      if 'has_gender_stereotype' in df.columns else 0)

    return df.dropna(subset=['lyrics_clean'])