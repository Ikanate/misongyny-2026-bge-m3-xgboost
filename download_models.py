from pathlib import Path
from huggingface_hub import snapshot_download, model_info

MODELS_DIR = Path("./DATA/models")
MODELS_DIR.mkdir(exist_ok=True)

MODELS = [
    {
        "repo_id": "cross-encoder/nli-MiniLM2-L6-H768",
        "local_dir": MODELS_DIR / "all-MiniLM-L6"
    },
    {
        "repo_id": "BAAI/bge-m3",
        "local_dir": MODELS_DIR / "bge-m3"
    }
]

for model in MODELS:
    local_dir = model["local_dir"]
    local_dir.mkdir(exist_ok=True)
    hash_file = local_dir / ".last_commit"

    remote_sha = model_info(model["repo_id"]).sha
    local_sha = hash_file.read_text().strip() if hash_file.exists() else None

    if local_sha == remote_sha:
        print(f"✓ {model['repo_id']} ya está actualizado ({remote_sha[:7]}).")
    else:
        print(f"Descargando {model['repo_id']}...")
        snapshot_download(
            repo_id=model["repo_id"],
            local_dir=str(local_dir),
            local_dir_use_symlinks=False
        )
        hash_file.write_text(remote_sha)
        print(f"✓ {model['repo_id']} descargado ({remote_sha[:7]}).")

print("Modelos listos.")