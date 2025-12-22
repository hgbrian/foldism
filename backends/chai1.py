"""Chai-1 structure prediction backend."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from modal import Image

from .common import (
    CACHE_VOLUME,
    LoggingOutput,
    MODEL_VOLUME,
    TIMEOUT,
    app,
    chai1_cache_key,
)

# Chai-1 needs more VRAM than L40S provides
CHAI1_GPU = "A100-80GB"

# =============================================================================
# Image
# =============================================================================

def _download_chai1_models():
    """Download Chai-1 models to volume during image build."""
    import os
    import torch
    from pathlib import Path
    from chai_lab.chai1 import run_inference

    os.environ["CHAI_DOWNLOADS_DIR"] = "/models"

    if Path("/models/_CHAI1_MODELS_DOWNLOADED").exists():
        print("[chai1] Models already downloaded")
        return

    print("[chai1] Downloading models...")
    Path("/tmp/in_dm").mkdir(parents=True, exist_ok=True)
    Path("/tmp/out_dm").mkdir(parents=True, exist_ok=True)
    with open("/tmp/in_dm/tmp.faa", "w") as f:
        f.write(">protein|name=pro\nMAWTPLLLLLLSHCTGSLSQPVLTQPTSL\n>ligand|name=lig\nCC\n")

    run_inference(
        fasta_file=Path("/tmp/in_dm/tmp.faa"),
        output_dir=Path("/tmp/out_dm"),
        num_trunk_recycles=1,
        num_diffn_timesteps=10,
        seed=1,
        device=torch.device("cuda:0"),
        use_esm_embeddings=True,
    )

    Path("/models/_CHAI1_MODELS_DOWNLOADED").touch()
    print("[chai1] Models downloaded")


chai1_image = (
    Image.debian_slim()
    .apt_install("git", "wget")
    .pip_install("polars==1.19.0", "requests", "urllib3")  # requests + deps
    .pip_install("git+https://github.com/chaidiscovery/chai-lab.git@af596cbc075a1fce368cec0ab5f31be1090ca7e2")
    .run_function(_download_chai1_models, gpu="A100-80GB", volumes={"/models": MODEL_VOLUME})
)

# =============================================================================
# Prediction
# =============================================================================


def _ensure_chai1_models():
    """Download Chai-1 models to volume if not present (runtime fallback)."""
    import torch
    from chai_lab.chai1 import run_inference

    MODEL_VOLUME.reload()

    marker = Path("/models/_CHAI1_MODELS_DOWNLOADED")
    if marker.exists():
        print("[chai1] Models already downloaded")
        return

    print("[chai1] Downloading models...")
    Path("/tmp/in_dm").mkdir(parents=True, exist_ok=True)
    Path("/tmp/out_dm").mkdir(parents=True, exist_ok=True)
    with open("/tmp/in_dm/tmp.faa", "w") as f:
        f.write(">protein|name=pro\nMAWTPLLLLLLSHCTGSLSQPVLTQPTSL\n>ligand|name=lig\nCC\n")

    os.environ["CHAI_DOWNLOADS_DIR"] = "/models"

    run_inference(
        fasta_file=Path("/tmp/in_dm/tmp.faa"),
        output_dir=Path("/tmp/out_dm"),
        num_trunk_recycles=1,
        num_diffn_timesteps=10,
        seed=1,
        device=torch.device("cuda:0"),
        use_esm_embeddings=True,
    )

    marker.touch()
    MODEL_VOLUME.commit()
    print("[chai1] Models downloaded")


@app.function(
    image=chai1_image,
    timeout=TIMEOUT * 60,
    gpu=CHAI1_GPU,
    volumes={"/models": MODEL_VOLUME, "/cache": CACHE_VOLUME},
)
def chai1_predict(params: dict[str, Any], overwrite: bool = False, job_id: str | None = None) -> list:
    """Run Chai-1 structure prediction."""
    import sys
    import torch
    from tempfile import TemporaryDirectory
    from chai_lab.chai1 import run_inference

    os.environ["CHAI_DOWNLOADS_DIR"] = "/models"
    os.environ["PYTHONUNBUFFERED"] = "1"

    # Set up logging early to capture all prints
    old_stdout, old_stderr = sys.stdout, sys.stderr
    logging_stdout = LoggingOutput(old_stdout, job_id, "chai1_logs")
    sys.stdout = sys.stderr = logging_stdout

    try:
        # Check models exist (read-only to allow parallel execution)
        _ensure_chai1_models()

        input_str = params["input_str"]
        input_name = params.get("input_name", "input.faa")
        num_trunk_recycles = params.get("num_trunk_recycles", 3)
        num_diffn_timesteps = params.get("num_diffn_timesteps", 200)
        seed = params.get("seed", 42)
        use_esm_embeddings = params.get("use_esm_embeddings", True)
        use_msa_server = params.get("use_msa_server", True)

        cache_key = chai1_cache_key(params)
        cache_path = Path(f"/cache/chai1/{cache_key}")
        cache_marker = cache_path / "_COMPLETE"

        if not overwrite and cache_marker.exists():
            print(f"[chai1] Cache HIT: {cache_key} (returning cached result)")
            CACHE_VOLUME.reload()
            logging_stdout.flush()
            sys.stdout, sys.stderr = old_stdout, old_stderr
            return [(f.relative_to(cache_path), f.read_bytes()) for f in cache_path.glob("**/*") if f.is_file() and f.name != "_COMPLETE"]

        print(f"[chai1] Cache MISS: {cache_key} (use_msa_server={use_msa_server})")

        with TemporaryDirectory() as in_dir, TemporaryDirectory() as out_dir:
            fasta_path = Path(in_dir) / input_name
            fasta_path.write_text(input_str)

            run_inference(
                fasta_file=fasta_path,
                output_dir=Path(out_dir),
                num_trunk_recycles=num_trunk_recycles,
                num_diffn_timesteps=num_diffn_timesteps,
                seed=seed,
                device=torch.device("cuda:0"),
                use_esm_embeddings=use_esm_embeddings,
                use_msa_server=use_msa_server,
            )

            outputs = [(f.relative_to(out_dir), f.read_bytes()) for f in Path(out_dir).rglob("*") if f.is_file()]

            cache_path.mkdir(parents=True, exist_ok=True)
            for rel_path, content in outputs:
                out_path = cache_path / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(content)

            cache_marker.touch()
            CACHE_VOLUME.commit()

            return outputs
    finally:
        logging_stdout.flush()
        sys.stdout, sys.stderr = old_stdout, old_stderr
