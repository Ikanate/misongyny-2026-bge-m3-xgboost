import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

def extract_embeddings_and_labels(model, dataloader, device):
    model.eval()
    all_embeddings = []
    all_labels_t1, all_labels_t2, all_labels_t3 = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            out    = model.backbone(input_ids=input_ids, attention_mask=attention_mask)
            pooled = mean_pooling(out, attention_mask)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

            all_embeddings.append(pooled.cpu().numpy())

            # ajusta los keys según tu __getitem__
            all_labels_t1.append(batch['label_t1'].numpy())
            all_labels_t2.append(batch['label_t2'].numpy())
            all_labels_t3.append(batch['label_t3'].numpy())

    return (
        np.vstack(all_embeddings),
        np.concatenate(all_labels_t1),
        np.concatenate(all_labels_t2),
        np.concatenate(all_labels_t3),
    )

def mean_pooling(model_output, attention_mask):

    token_embeddings = model_output.last_hidden_state          # (B, L, H)
    mask_expanded    = attention_mask.unsqueeze(-1).float()    # (B, L, 1)
    sum_embeddings   = (token_embeddings * mask_expanded).sum(1)  # (B, H)
    sum_mask         = mask_expanded.sum(1).clamp(min=1e-9)       # (B, 1)
    return sum_embeddings / sum_mask                              # (B, H)