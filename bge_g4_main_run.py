from train.train_code_pl import *
from pytorch_lightning.callbacks import ModelCheckpoint

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

#ML Modules
import train.train_code as tr_c
import preprocess.data_frame_preprocess as prep
import models.model_bge_m3 as bge_m3
import models.model_xgboost as xgb_m3
import evaluate.long_tokens
import evaluate.long_tokens as long_tok
import dataset.Train as multi_task

#Config FILE
import config as cfg

_VALID_METRICS = {'f1_t1', 'f1_t2', 'f1_t3', 'avg_f1'}


cfg.train.LOSS_W1 = 1.0
cfg.train.LOSS_W2 = 0.0
cfg.train.LOSS_W3 = 0.0

cfg.train.BEST_MODEL_METRIC  = 'f1_t1'
cfg.train.GRADIENT_CHECKPOINTING = False
cfg.train.BATCH_SIZE   = 4
cfg.train.EPOCHS = 15
cfg.train.SEED = 30


os.makedirs(cfg.paths.SAVE_DIR,exist_ok=True)

#dataset
raw_df = pd.read_csv(cfg.paths.CSV_PATH)
df = prep.prepare_dataframe(raw_df,cfg) #New columns: lyrics_clean & gloss_clean
strat_key = (df['label_t1'].astype(str) + '_' +
             df['label_t3'].astype(str) + '_' +
             df[cfg.columns.LABEL_COLS_T2].astype(str).agg('-'.join, axis=1))


#Load Transformer Model
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer,model = bge_m3.load_tokenizer_model(cfg,device)


# Generate DataSets and DataLoaders
train_df, val_df = train_test_split(df, test_size=cfg.train.TEST_SIZE,
                                    stratify=strat_key,
                                    random_state= cfg.train.SEED)

#Dataset
train_ds = multi_task.MultiTaskDataset(train_df, cfg,tokenizer, cfg.tokenization.MAX_LEN,
                            head_len=cfg.tokenization.HEAD_LEN,
                            model_prefix=cfg.paths.MODEL_PREFIX)
val_ds   = multi_task.MultiTaskDataset(val_df, cfg,  tokenizer, cfg.tokenization.MAX_LEN,
                            head_len=cfg.tokenization.HEAD_LEN,
                            model_prefix=cfg.paths.MODEL_PREFIX)
#DataLoader
train_loader = DataLoader(train_ds, batch_size=cfg.train.BATCH_SIZE, shuffle=True,
                          num_workers=6, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=cfg.train.BATCH_SIZE,
                          num_workers=6, pin_memory=True)


# model, tokenizer, cfg, train_df, train_loader, val_loader vienen de tu setup
lit = MultiTaskLitModule(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        train_df=train_df,
        t2_label_cols=['type_sexualization', 'type_violence', 'type_hate', 'nothing'],
    )

trainer = build_trainer(cfg)

trainer.fit(lit, 
            train_dataloaders=train_loader, 
            val_dataloaders=val_loader)


print('Mejor ckpt:', trainer.checkpoint_callback.best_model_path)
print('Mejor métrica:', trainer.checkpoint_callback.best_model_score)