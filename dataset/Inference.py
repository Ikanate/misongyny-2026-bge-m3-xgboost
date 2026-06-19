from torch.utils.data import Dataset, DataLoader

class MultiTaskDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer,
                 max_len: int = 512, head_len=False,
                 model_prefix: str = ''):
        
        self.texts      = df['lyrics_clean'].tolist()
        # ── NUEVO: glosa opcional ──────────────────────────────────────────
        self.glosses    = df['gloss_clean'].tolist() if 'gloss_clean' in df.columns else [''] * len(df)
        self.use_gloss  = cfg.USE_GLOSS
        self.gloss_sep  = cfg.GLOSS_SEP
        # ──────────────────────────────────────────────────────────────────
        self.label_t1   = df['label_t1'].tolist()
        self.label_t2   = df[cfg.LABEL_COLS_T2].values.astype(float)
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




class InferenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int, head_len,
                 model_prefix: str = ''):
        self.texts      = df['lyrics_clean'].tolist()
        # ── NUEVO ──────────────────────────────────────────────────────────
        self.glosses    = df['gloss_clean'].tolist() if 'gloss_clean' in df.columns else [''] * len(df)
        self.use_gloss  = cfg.USE_GLOSS
        # ──────────────────────────────────────────────────────────────────
        self.tokenizer  = tokenizer
        self.max_len    = max_len
        self.head_len   = head_len
        self.prefix_ids = tokenizer.encode(model_prefix, add_special_tokens=False) if model_prefix else []
        
        self.max_text_tokens = max_len - len(self.prefix_ids) - 2
        self.tail_len   = (self.max_text_tokens - head_len) if head_len is not False else None
        self.sep_ids    = [tokenizer.sep_token_id] if self.use_gloss else []
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text_tokens  = self.tokenizer.encode(str(self.texts[idx]), add_special_tokens=False)
        glosa        = str(self.glosses[idx]) if self.use_gloss else ''

        if self.use_gloss and glosa.strip():
            gloss_tokens = self.tokenizer.encode(glosa, add_special_tokens=False)
            budget        = self.max_text_tokens - len(self.sep_ids)
            lyrics_budget = int(budget * 0.90)
            gloss_budget  = budget - lyrics_budget

            if len(text_tokens) > lyrics_budget:
                if self.head_len is False:
                    text_tokens = text_tokens[:lyrics_budget]
                else:
                    hl = min(self.head_len, lyrics_budget - 1)
                    tl = lyrics_budget - hl
                    text_tokens = text_tokens[:hl] + text_tokens[-tl:]

            gloss_tokens   = gloss_tokens[:gloss_budget]
            combined_tokens = text_tokens + self.sep_ids + gloss_tokens
        else:
            if len(text_tokens) > self.max_text_tokens:
                if self.head_len is False:
                    text_tokens = text_tokens[:self.max_text_tokens]
                else:
                    text_tokens = text_tokens[:self.head_len] + text_tokens[-self.tail_len:]
            combined_tokens = text_tokens

        cls_id         = self.tokenizer.cls_token_id
        sep_id         = self.tokenizer.sep_token_id
        input_ids      = [cls_id] + self.prefix_ids + combined_tokens + [sep_id]
        attention_mask = [1] * len(input_ids)

        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids      += [self.tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len

        return {
            'input_ids':      torch.tensor(input_ids,      dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }



def predict_dataset(df: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
    """
    Inferencia sobre df completo usando DataLoader.
    Devuelve df original con columnas de predicción añadidas.
    """
    infer_ds     = InferenceDataset(df, tokenizer, cfg.MAX_LEN, cfg.HEAD_LEN,
                                    model_prefix=cfg.MODEL_PREFIX)
    infer_loader = DataLoader(infer_ds, batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    all_t1_label, all_t1_prob_M, all_t1_prob_NM = [], [], []
    all_t2_scores = {col: [] for col in cfg.LABEL_COLS_T2}
    all_t2_active = []
    all_t3_label, all_t3_prob_Y, all_t3_prob_N  = [], [], []

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(infer_loader):
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)

            lg1, lg2, lg3 = model(ids, mask)

            p1 = torch.softmax(lg1, dim=-1)
            all_t1_prob_NM.extend(p1[:, 0].cpu().tolist())
            all_t1_prob_M.extend( p1[:, 1].cpu().tolist())
            all_t1_label.extend(['M' if v > 0.5 else 'NM'
                                  for v in p1[:, 1].cpu().tolist()])

            p2    = torch.sigmoid(lg2)
            flags = (p2 >= cfg.THRESHOLD_T2).int().cpu().tolist()
            scores = p2.cpu().tolist()
            for j, col in enumerate(cfg.LABEL_COLS_T2):
                all_t2_scores[col].extend([round(s[j], 4) for s in scores])
            for row_flags in flags:
                active = [cfg.LABEL_COLS_T2[j]
                          for j, f in enumerate(row_flags) if f]
                all_t2_active.append('|'.join(active) or 'nothing')

            p3 = torch.softmax(lg3, dim=-1)
            all_t3_prob_N.extend(p3[:, 0].cpu().tolist())
            all_t3_prob_Y.extend(p3[:, 1].cpu().tolist())
            all_t3_label.extend(['Y' if v > 0.5 else 'N'
                                  for v in p3[:, 1].cpu().tolist()])

            print(f'  Batch {i+1}/{len(infer_loader)} '
                  f'({min((i+1)*batch_size, len(df))}/{len(df)} canciones)…')

    preds = pd.DataFrame({
        'pred_t1_label':   all_t1_label,
        'pred_t1_prob_M':  [round(v, 4) for v in all_t1_prob_M],
        'pred_t1_prob_NM': [round(v, 4) for v in all_t1_prob_NM],
        **{f'pred_t2_{col.replace("type_", "")}': all_t2_scores[col]
           for col in cfg.LABEL_COLS_T2},
        'pred_t2_active':  all_t2_active,
        'pred_t3_label':   all_t3_label,
        'pred_t3_prob_Y':  [round(v, 4) for v in all_t3_prob_Y],
        'pred_t3_prob_N':  [round(v, 4) for v in all_t3_prob_N],
    })

    return pd.concat([df.reset_index(drop=True), preds], axis=1)

