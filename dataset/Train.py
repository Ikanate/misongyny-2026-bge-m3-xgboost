from torch.utils.data import Dataset
import pandas as pd
import torch


class MultiTaskDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame, cfg, tokenizer,
                 max_len: int = 512, head_len=False,
                 model_prefix: str = ''):
        
        self.texts      = df['lyrics_clean'].tolist()
        # ── NUEVO: glosa opcional ──────────────────────────────────────────
        self.glosses    = df['gloss_clean'].tolist() if 'gloss_clean' in df.columns else [''] * len(df)
        
        self.use_gloss  = cfg.flags.USE_GLOSS
        self.gloss_sep  = cfg.flags.GLOSS_SEP
        # ──────────────────────────────────────────────────────────────────
        self.label_t1   = df['label_t1'].tolist()
        self.label_t2   = df[cfg.columns.LABEL_COLS_T2].values.astype(float)
        self.label_t3   = df['label_t3'].tolist()
        self.tokenizer  = tokenizer
        self.max_len    = max_len
        self.head_len   = head_len
        self.prefix     = f''

        self.prefix_ids      = tokenizer.encode(self.prefix, add_special_tokens=False) if self.prefix else []
        
        self.max_text_tokens = max_len - len(self.prefix_ids) - 2

        if head_len is not False:
            self.tail_len = self.max_text_tokens - head_len
        else:
            self.tail_len = None

        # ── NUEVO: tokens del separador (pre-calculados una sola vez) ──────
        self.sep_ids = [tokenizer.sep_token_id]

    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        texto = str(self.texts[idx])
        glosa = str(self.glosses[idx]) if self.use_gloss else ''

        # ── NUEVO: construir texto base (letra + glosa si aplica) ──────────
        text_tokens = self.tokenizer.encode(texto, add_special_tokens=False)

        if self.use_gloss and glosa.strip():
            gloss_tokens = self.tokenizer.encode(glosa, add_special_tokens=False)
            # presupuesto: max_text_tokens - len(sep_ids)
            budget = self.max_text_tokens - len(self.sep_ids)
            # repartir budget 70% letra / 30% glosa (ajustable)
            lyrics_budget = int(budget * 0.90)
            gloss_budget  = budget - lyrics_budget

            if len(text_tokens) > lyrics_budget:
                if self.head_len is False:
                    text_tokens = text_tokens[:lyrics_budget]
                else:
                    # head+tail dentro del presupuesto de letra
                    hl = min(self.head_len, lyrics_budget - 1)
                    tl = lyrics_budget - hl
                    text_tokens = text_tokens[:hl] + text_tokens[-tl:]

            gloss_tokens = gloss_tokens[:gloss_budget]
            combined_tokens = text_tokens + self.sep_ids + gloss_tokens
        else:
            # ── sin glosa: lógica original intacta ────────────────────────
            if len(text_tokens) > self.max_text_tokens:
                if self.head_len is False:
                    text_tokens = text_tokens[:self.max_text_tokens]
                else:
                    text_tokens = (text_tokens[:self.head_len] +
                                   text_tokens[-self.tail_len:])
            combined_tokens = text_tokens
        # ──────────────────────────────────────────────────────────────────

        cls_id         = self.tokenizer.cls_token_id
        sep_id         = self.tokenizer.sep_token_id
        input_ids      = [cls_id] + self.prefix_ids + combined_tokens + [sep_id]
        attention_mask = [1] * len(input_ids)

        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            pad_id         = self.tokenizer.pad_token_id
            input_ids      = input_ids      + [pad_id] * pad_len
            attention_mask = attention_mask + [0]      * pad_len

        return {
            'input_ids':      torch.tensor(input_ids,      dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'label_t1': torch.tensor(self.label_t1[idx],  dtype=torch.long),
            'label_t2': torch.tensor(self.label_t2[idx],  dtype=torch.float),
            'label_t3': torch.tensor(self.label_t3[idx],  dtype=torch.long),
        }
