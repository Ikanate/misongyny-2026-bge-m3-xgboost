"""
config.py
══════════════════════════════════════════════════════════════════════════════
Configuración centralizada para el entrenamiento multi-task.
Modifica este archivo para ajustar rutas e hiperparámetros sin tocar el
script principal.
══════════════════════════════════════════════════════════════════════════════
"""

# ── Rutas ──────────────────────────────────────────────────────────────────
#W GLOSSES Documents/MiSongGyny2026/src/ollama_modules/resultado.csv

#Public Test W Glosses
#/home/andreoseo/Documents/MiSongGyny2026/DATA/test/w_glossa/resultado_public_test.csv
class paths:
    #CSV_PATH     = '/home/andreoseo/Documents/MiSongGyny2026/src/ollama_modules/resultado.csv' #'../../DATA/dataset_limpio.csv'#
    CSV_PATH         = "/home/andreoseo/Documents/MiSongGyny2026/src/ollama_modules/resultado_manual_gloss_clean.csv"
    AUG_CSV_PATH     = '../DATA/train_pahse_misongyny_26_train_df_aug.csv'

    # Ruta al modelo descargado localmente 
    MODEL_DIR        = '/home/andreoseo/Documents/MiSongGyny2026/modelos/bge-m3-base'

    SAVE_DIR         = './checkpoints/' #bg3_deduplicatedn_t1
    SAVE_DIR_XGBOOST = './checkpoints/xgboost/'
    MODEL_PREFIX     = ''

# ── Tokenización ───────────────────────────────────────────────────────────
class tokenization:
    MAX_LEN  = 1024
    HEAD_LEN = False
    # HEAD_LEN: número de tokens tomados desde el inicio antes del corte.
    #   - Entero  → activa estrategia Head + Tail  (ej. 256)
    #   - False   → trunca solo desde el principio (hasta MAX_LEN tokens)

class train:
    # ── Entrenamiento
    BATCH_SIZE   = 4
    EPOCHS       = 10
    LR_BACKBONE  = 5e-6
    LR_HEADS     = 1e-3
    WEIGHT_DECAY = 0.1
    WARMUP_RATIO = 0.1

    # ── Pesos de la loss combinada 
    LOSS_W1 = 1.0   # Task 1 – misoginia binaria
    LOSS_W2 = 0.0   # Task 2 – tipo multilabel
    LOSS_W3 = 0.0   # Task 3 – estereotipo binario


    # ── Inferencia
    THRESHOLD_T2 = 0.5   # umbral sigmoid para Task 2

    # -- Val/Train Dataset
    TEST_SIZE     = 0.3
    SEED = 42
    
    BEST_MODEL_METRIC = 'f1_t1'
    GRADIENT_CHECKPOINTING = False

    PATIENCE = 3
    MIN_DELTA = 0.001

    

# ── Columnas multilabel (Task 2) ───────────────────────────────────────────
class columns:
    LABEL_COLS_T2 = ['type_sexualization', 'type_violence', 'type_hate', 'nothing']
    GLOSS_COL = 'resultado' # Name col in CSV
    LYRICS_COL = 'lyrics'

class flags:
    USE_AUGMENTED = False
    USE_GLOSS = False
    GLOSS_SEP = ' [SEP] '


