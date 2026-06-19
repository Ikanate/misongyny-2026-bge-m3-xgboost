import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

# ── 1. AUTOCAST: aprovecha Tensor Cores (FP16/BF16) en la VRAM ──────────────
#    Sin esto, todo corre en FP32 y usas el doble de memoria.
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state          # (B, L, H)
    mask_expanded    = attention_mask.unsqueeze(-1).float()    # (B, L, 1)
    sum_embeddings   = (token_embeddings * mask_expanded).sum(1)
    sum_mask         = mask_expanded.sum(1).clamp(min=1e-9)
    return sum_embeddings / sum_mask                           # (B, H)


def extract_embeddings_and_labels(model, dataloader, device):
    model.eval()

    all_embeddings = []
    all_labels_t1, all_labels_t2, all_labels_t3 = [], [], []

    # ── 2. CUDA STREAMS: overlap entre transferencia CPU↔GPU y cómputo ───────
    stream = torch.cuda.Stream() if device.type == "cuda" else None

    with torch.no_grad():
        for batch in dataloader:

            # ── 3. PIN MEMORY + non_blocking: la transferencia no bloquea CPU ─
            input_ids      = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)

            with torch.cuda.stream(stream) if stream else torch.no_grad():
                # ── 4. AUTOCAST: FP16/BF16 automático dentro del forward ─────
                with torch.autocast(device_type=device.type, dtype=AMP_DTYPE,
                                    enabled=(device.type == "cuda")):
                    out    = model.backbone(input_ids=input_ids,
                                           attention_mask=attention_mask)
                    pooled = mean_pooling(out, attention_mask)

                # Normalize en FP32 para estabilidad numérica
                pooled = torch.nn.functional.normalize(
                    pooled.float(), p=2, dim=1
                )

            # ── 5. Sincronizar stream antes de bajar a CPU ────────────────────
            if stream:
                stream.synchronize()

            # ── 6. Bajar a CPU con numpy() directo (evita copia extra) ────────
            all_embeddings.append(pooled.cpu().numpy())
            all_labels_t1.append(batch['label_t1'].numpy())
            all_labels_t2.append(batch['label_t2'].numpy())
            all_labels_t3.append(batch['label_t3'].numpy())

    return (
        np.vstack(all_embeddings),
        np.concatenate(all_labels_t1),
        np.concatenate(all_labels_t2),
        np.concatenate(all_labels_t3),
    )


# ── 7. DataLoader optimizado: workers + pin_memory ───────────────────────────
def make_dataloader(dataset: Dataset, batch_size: int = 64) -> DataLoader:
    import os
    num_workers = min(8, os.cpu_count() or 1)   # no más workers que cores

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,       # carga paralela en CPU
        pin_memory=True,               # tensores en memoria pinned → DMA directo a GPU
        pin_memory_device="cuda",      # PyTorch ≥ 2.0
        prefetch_factor=2,             # 2 batches pre-cargados por worker
        persistent_workers=True,       # workers viven entre epochs (evita fork overhead)
    )