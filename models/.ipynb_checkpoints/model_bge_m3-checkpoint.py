from torch import nn
from transformers import (AutoTokenizer, AutoModel,
                          get_linear_schedule_with_warmup)
import torch.nn.functional as F


class MultiTaskBGE(nn.Module):
    """
            Backbone BGE-M3 
                + 
            3 Head Classification
    """
    def __init__(self, model_dir: str, dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_dir)   # sin trust_remote_code
        hidden = self.backbone.config.hidden_size   # 1024 para BGE-M3

        self.dropout = nn.Dropout(dropout)
        self.head_t1 = nn.Linear(hidden, 2)
        self.head_t2 = nn.Linear(hidden, 4)
        self.head_t3 = nn.Linear(hidden, 2)

        # Gradient checkpointing (opcional, ayuda con VRAM)
        #self.backbone.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pooling(out, attention_mask)
        pooled = F.normalize(pooled, p=2, dim=1)
        pooled = self.dropout(pooled)
        return self.head_t1(pooled), self.head_t2(pooled), self.head_t3(pooled)
        

def mean_pooling(model_output, attention_mask):

    token_embeddings = model_output.last_hidden_state          # (B, L, H)
    mask_expanded    = attention_mask.unsqueeze(-1).float()    # (B, L, 1)
    sum_embeddings   = (token_embeddings * mask_expanded).sum(1)  # (B, H)
    sum_mask         = mask_expanded.sum(1).clamp(min=1e-9)       # (B, 1)
    return sum_embeddings / sum_mask                              # (B, H)




def load_tokenizer_model(cfg, device):
    tokenizer = AutoTokenizer.from_pretrained(cfg.paths.MODEL_DIR)
    model     = MultiTaskBGE(cfg.paths.MODEL_DIR).to(device)
    return tokenizer, model

    

