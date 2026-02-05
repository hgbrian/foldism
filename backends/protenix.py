"""Protenix structure prediction backend."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from modal import Image

from .common import (
    CACHE_VOLUME,
    GPU,
    MODEL_VOLUME,
    TIMEOUT,
    _fasta_to_protenix_json,
    app,
    job_store,
    protenix_cache_key,
    write_log_line,
)

# =============================================================================
# Image
# =============================================================================


def _setup_protenix_volume(prefix: str = "[protenix]") -> Path:
    """Set up symlink from site-packages to volume. Returns volume_data path."""
    import shutil
    import site

    site_pkg = Path(site.getsitepackages()[0])
    release_data = site_pkg / "release_data"
    volume_data = Path("/models/protenix_data")

    if not release_data.is_symlink():
        if release_data.exists():
            shutil.rmtree(release_data)
        volume_data.mkdir(parents=True, exist_ok=True)
        release_data.symlink_to(volume_data)
        print(f"{prefix} Symlinked {release_data} -> {volume_data}")

    return volume_data


def _download_protenix_file(filepath: Path, url: str, prefix: str = "[protenix]") -> bool:
    """Download a file if it doesn't exist. Returns True if downloaded."""
    import subprocess

    if filepath.exists():
        print(f"{prefix} Already exists: {filepath.name} ({filepath.stat().st_size:,} bytes)")
        return False

    filepath.parent.mkdir(parents=True, exist_ok=True)
    print(f"{prefix} Downloading {filepath.name}...")
    subprocess.run(f'wget -q -O "{filepath}" "{url}"', shell=True, check=True)

    if not filepath.exists():
        raise RuntimeError(f"Download failed: {filepath}")
    print(f"{prefix} Downloaded: {filepath.name} ({filepath.stat().st_size:,} bytes)")
    return True


def _download_protenix_models():
    """Download protenix models and common files to VOLUME during image build."""
    from protenix.web_service.dependency_url import URL as PROTENIX_URLS

    volume_data = _setup_protenix_volume(prefix="[BUILD]")
    checkpoint_dir = volume_data / "checkpoint"
    ccd_dir = volume_data / "ccd_cache"

    downloads = [
        (checkpoint_dir / "protenix_base_default_v1.0.0.pt", PROTENIX_URLS["protenix_base_default_v1.0.0"]),
        (checkpoint_dir / "protenix_mini_default_v0.5.0.pt", PROTENIX_URLS["protenix_mini_default_v0.5.0"]),
        (ccd_dir / "components.cif", PROTENIX_URLS["ccd_components_file"]),
        (ccd_dir / "components.cif.rdkit_mol.pkl", PROTENIX_URLS["ccd_components_rdkit_mol_file"]),
        (ccd_dir / "clusters-by-entity-40.txt", PROTENIX_URLS["pdb_cluster_file"]),
    ]

    for filepath, url in downloads:
        _download_protenix_file(filepath, url, prefix="[BUILD]")


protenix_image = (
    Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "wget", "clang")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install("polars==1.19.0", "protenix==1.0.0")
    .run_function(_download_protenix_models, gpu="L40S", volumes={"/models": MODEL_VOLUME})
)

# =============================================================================
# Model Download
# =============================================================================


def _ensure_protenix_models(model: str = "protenix"):
    """Verify Protenix models exist on volume (should be pre-downloaded at image build)."""
    MODEL_VOLUME.reload()
    volume_data = _setup_protenix_volume()

    if model == "protenix_mini":
        checkpoint_file = volume_data / "checkpoint" / "protenix_mini_default_v0.5.0.pt"
    else:
        checkpoint_file = volume_data / "checkpoint" / "protenix_base_default_v1.0.0.pt"

    if checkpoint_file.exists():
        print(f"[protenix] Model found: {checkpoint_file.name}")
        return

    raise RuntimeError(
        f"Model not found: {checkpoint_file}. "
        "Rebuild with: uv run modal run --force-build foldism.py"
    )


# =============================================================================
# Prediction
# =============================================================================


@app.function(
    image=protenix_image,
    timeout=TIMEOUT * 60,
    gpu=GPU,
    volumes={"/models": MODEL_VOLUME, "/cache": CACHE_VOLUME},
)
def protenix_predict(params: dict[str, Any], overwrite: bool = False, job_id: str | None = None) -> list:
    """Run Protenix structure prediction."""
    import subprocess
    from tempfile import TemporaryDirectory

    input_str = params["input_str"]
    input_name = params.get("input_name", "input")
    seeds = params.get("seeds", "42")
    use_msa = params.get("use_msa", True)
    model = params.get("model", "protenix")  # "protenix" or "protenix_mini"

    # Determine log key based on model
    log_key = "protenix_logs" if model == "protenix" else "protenix_mini_logs"

    # Check models exist (read-only to allow parallel execution)
    write_log_line(job_id, log_key, f"[{model}] Checking models...", model)
    _ensure_protenix_models(model)

    cache_key = protenix_cache_key(params)
    cache_path = Path(f"/cache/protenix/{cache_key}")
    cache_marker = cache_path / "_COMPLETE"

    if not overwrite and cache_marker.exists():
        msg = f"[{model}] Cache HIT: {cache_key} (returning cached result)"
        print(msg)
        write_log_line(job_id, log_key, msg, model)
        CACHE_VOLUME.reload()
        return [(f.relative_to(cache_path), f.read_bytes()) for f in cache_path.glob("**/*") if f.is_file() and f.name != "_COMPLETE"]

    msg = f"[{model}] Cache MISS: {cache_key} (use_msa={use_msa})"
    print(msg)
    write_log_line(job_id, log_key, msg, model)

    if input_str.strip().startswith(">"):
        input_str = _fasta_to_protenix_json(input_str, input_name)

    with TemporaryDirectory() as in_dir, TemporaryDirectory() as out_dir:
        json_path = Path(in_dir) / "input.json"
        json_path.write_text(input_str)

        use_msa_str = "true" if use_msa else "false"
        model_name = "protenix_mini_default_v0.5.0" if model == "protenix_mini" else "protenix_base_default_v1.0.0"
        cmd = f'stdbuf -oL protenix pred --input "{json_path}" --out_dir "{out_dir}" --seeds {seeds} --use_msa {use_msa_str} --model_name "{model_name}"'
        print(f"Running: {cmd}")
        write_log_line(job_id, log_key, f"Command: {cmd}", model)

        # Use Popen to capture output in real-time with batched log writes
        import time
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )

        # Stream output with batched log writes (use readline to avoid iterator buffering)
        log_buffer = []
        last_flush = time.time()
        FLUSH_INTERVAL = 10.0

        def flush_logs():
            nonlocal log_buffer, last_flush
            if log_buffer and job_id:
                try:
                    existing = job_store.get(job_id, {})
                    logs = existing.get(log_key, [])
                    logs.extend(log_buffer)
                    existing[log_key] = logs[-1000:]
                    job_store[job_id] = existing
                except Exception as e:
                    print(f"[{model}] Failed to flush logs: {e}")
                log_buffer = []
            last_flush = time.time()

        while True:
            line = process.stdout.readline()
            if not line:  # EOF
                break
            line = line.rstrip()
            if line:
                print(line, flush=True)
                log_buffer.append(line)
                if time.time() - last_flush >= FLUSH_INTERVAL:
                    flush_logs()

        flush_logs()  # Flush remaining
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

        pred_dir = Path(out_dir) / input_name
        if not pred_dir.exists():
            pred_dir = Path(out_dir)

        outputs = [(f.relative_to(pred_dir), f.read_bytes()) for f in pred_dir.rglob("*") if f.is_file()]

        cache_path.mkdir(parents=True, exist_ok=True)
        for rel_path, content in outputs:
            out_path = cache_path / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(content)

        cache_marker.touch()
        CACHE_VOLUME.commit()

        return outputs
