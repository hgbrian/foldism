"""Upload Protenix's shared (non-checkpoint) files from the Modal model volume
to a HuggingFace bucket, so we can replace the ByteDance URLs with HF mirrors.

Files uploaded (everything in /models/protenix_data/ccd_cache/ EXCEPT the
model `.pt`):
    - components.cif
    - components.cif.rdkit_mol.pkl
    - clusters-by-entity-40.txt
    - obsolete_release_date.csv

Usage:
    # 1. Create a Modal secret with your HF write token
    uvx modal secret create huggingface HF_TOKEN=hf_xxxxxxxxxxxxx

    # 2. Run this script
    uvx modal run scripts/upload_protenix_shared.py

The bucket repo_id is btnaughton/bm9pc2VkcmF3 (the same bucket that already
holds protenix-v2.pt at 1.86 GB).
"""

from __future__ import annotations

import os
from pathlib import Path

from modal import App, Image, Secret, Volume

app = App("foldism-upload-protenix-shared")

MODEL_VOLUME = Volume.from_name("foldism-models", create_if_missing=False)

upload_image = Image.debian_slim(python_version="3.12").pip_install("huggingface_hub>=0.27")

BUCKET_REPO_ID = "btnaughton/bm9pc2VkcmF3"
SHARED_FILES = [
    "components.cif",
    "components.cif.rdkit_mol.pkl",
    "clusters-by-entity-40.txt",
    "obsolete_release_date.csv",
]


@app.function(
    image=upload_image,
    volumes={"/models": MODEL_VOLUME},
    secrets=[Secret.from_name("huggingface")],
    timeout=3600,
)
def upload() -> list[str]:
    """Upload SHARED_FILES from the volume to the HF bucket. Returns uploaded names."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN env var not found. Add it to the 'huggingface' Modal secret."
        )

    api = HfApi(token=token)
    ccd_dir = Path("/models/protenix_data/ccd_cache")

    uploaded: list[str] = []
    for name in SHARED_FILES:
        src = ccd_dir / name
        if not src.exists():
            print(f"  SKIP {name} (not found at {src})")
            continue
        size_mb = src.stat().st_size / 1e6
        print(f"  Uploading {name} ({size_mb:.1f} MB) → {BUCKET_REPO_ID}/{name}...")
        # repo_type="model" matches the bucket's underlying repo. If your HF
        # bucket uses a different type ("dataset" or a bespoke "bucket" value),
        # update here.
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=name,
            repo_id=BUCKET_REPO_ID,
            repo_type="model",
        )
        uploaded.append(name)
        print(f"  Done.")

    print(f"\nUploaded {len(uploaded)}/{len(SHARED_FILES)} files to {BUCKET_REPO_ID}")
    return uploaded


@app.local_entrypoint()
def main():
    uploaded = upload.remote()
    print()
    print("Done. Files in bucket:")
    for n in uploaded:
        print(f"  - https://huggingface.co/{BUCKET_REPO_ID}/resolve/main/{n}")
