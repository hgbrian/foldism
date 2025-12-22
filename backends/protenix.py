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


def _download_protenix_base():
    """Download protenix base model to VOLUME during image build."""
    import shutil
    import site
    import subprocess
    from pathlib import Path

    # Set up symlink from site-packages to volume
    site_pkg = Path(site.getsitepackages()[0])
    release_data = site_pkg / "release_data"
    volume_data = Path("/models/protenix_data")

    if not release_data.is_symlink():
        if release_data.exists():
            shutil.rmtree(release_data)
        volume_data.mkdir(parents=True, exist_ok=True)
        release_data.symlink_to(volume_data)
        print(f"[BUILD] Symlinked {release_data} -> {volume_data}")

    checkpoint_file = volume_data / "checkpoint" / "protenix_base_default_v0.5.0.pt"

    if checkpoint_file.exists():
        size = checkpoint_file.stat().st_size
        print(f"[BUILD] Protenix base model already exists ({size:,} bytes)")
        return

    print("[BUILD] Downloading protenix base model to VOLUME...")
    test_json = '[{"name": "test", "sequences": [{"proteinChain": {"sequence": "MAWTPLLLLLLSHCTGSLSQPVLTQPTSL", "count": 1}}]}]'
    Path("/tmp/in_dm").mkdir(parents=True, exist_ok=True)
    Path("/tmp/out_dm").mkdir(parents=True, exist_ok=True)
    Path("/tmp/in_dm/test.json").write_text(test_json)

    subprocess.run(
        "protenix predict --input /tmp/in_dm/test.json --out_dir /tmp/out_dm --seeds 42 --use_msa false",
        shell=True,
        check=True,
    )

    if checkpoint_file.exists():
        size = checkpoint_file.stat().st_size
        print(f"[BUILD] Protenix base model downloaded successfully ({size:,} bytes)")
    else:
        raise RuntimeError("Protenix base model download failed!")


def _download_protenix_mini():
    """Download protenix-mini model to VOLUME during image build."""
    import shutil
    import site
    import subprocess
    from pathlib import Path

    # Set up symlink from site-packages to volume
    site_pkg = Path(site.getsitepackages()[0])
    release_data = site_pkg / "release_data"
    volume_data = Path("/models/protenix_data")

    if not release_data.is_symlink():
        if release_data.exists():
            shutil.rmtree(release_data)
        volume_data.mkdir(parents=True, exist_ok=True)
        release_data.symlink_to(volume_data)
        print(f"[BUILD] Symlinked {release_data} -> {volume_data}")

    checkpoint_file = volume_data / "checkpoint" / "protenix_mini_default_v0.5.0.pt"

    if checkpoint_file.exists():
        size = checkpoint_file.stat().st_size
        print(f"[BUILD] Protenix-mini model already exists ({size:,} bytes)")
        return

    print("[BUILD] Downloading protenix-mini model to VOLUME...")
    test_json = '[{"name": "test", "sequences": [{"proteinChain": {"sequence": "MAWTPLLLLLLSHCTGSLSQPVLTQPTSL", "count": 1}}]}]'
    Path("/tmp/in_dm").mkdir(parents=True, exist_ok=True)
    Path("/tmp/out_dm").mkdir(parents=True, exist_ok=True)
    Path("/tmp/in_dm/test.json").write_text(test_json)

    subprocess.run(
        'protenix predict --input /tmp/in_dm/test.json --out_dir /tmp/out_dm --seeds 42 --use_msa false --model_name "protenix_mini_default_v0.5.0"',
        shell=True,
        check=True,
    )

    if checkpoint_file.exists():
        size = checkpoint_file.stat().st_size
        print(f"[BUILD] Protenix-mini model downloaded successfully ({size:,} bytes)")
    else:
        raise RuntimeError("Protenix-mini model download failed!")


protenix_image = (
    Image.debian_slim()
    .apt_install("git", "wget")
    .pip_install("polars==1.19.0", "protenix==0.6.1")
    .run_function(_download_protenix_base, gpu="L40S", volumes={"/models": MODEL_VOLUME})
    .run_function(_download_protenix_mini, gpu="L40S", volumes={"/models": MODEL_VOLUME})
)

# =============================================================================
# Model Download
# =============================================================================


def _ensure_protenix_models(model: str = "protenix"):
    """Ensure Protenix models exist on volume (download if missing)."""
    import shutil
    import site
    import subprocess

    # Reload volume to see latest data
    MODEL_VOLUME.reload()

    # Set up symlink from site-packages to volume
    site_pkg = Path(site.getsitepackages()[0])
    release_data = site_pkg / "release_data"
    volume_data = Path("/models/protenix_data")

    if not release_data.is_symlink():
        if release_data.exists():
            if not volume_data.exists():
                shutil.move(str(release_data), str(volume_data))
            else:
                shutil.rmtree(release_data)
        volume_data.mkdir(parents=True, exist_ok=True)
        release_data.symlink_to(volume_data)
        print(f"[protenix] Symlinked {release_data} -> {volume_data}")

    if model == "protenix_mini":
        checkpoint_file = volume_data / "checkpoint" / "protenix_mini_default_v0.5.0.pt"
        model_name = "protenix_mini_default_v0.5.0"
    else:
        checkpoint_file = volume_data / "checkpoint" / "protenix_base_default_v0.5.0.pt"
        model_name = None

    # Check if model exists on volume
    if checkpoint_file.exists():
        print(f"[protenix] Model found on volume: {model}")
        return

    # Download model to volume if missing
    print(f"[protenix] Model not on volume, downloading: {model}")
    Path("/tmp/in_dm").mkdir(parents=True, exist_ok=True)
    Path("/tmp/out_dm").mkdir(parents=True, exist_ok=True)
    test_json = '[{"name": "test", "sequences": [{"proteinChain": {"sequence": "MAWTPLLLLLLSHCTGSLSQPVLTQPTSL", "count": 1}}]}]'
    Path("/tmp/in_dm/test.json").write_text(test_json)

    cmd = "protenix predict --input /tmp/in_dm/test.json --out_dir /tmp/out_dm --seeds 42 --use_msa false"
    if model_name:
        cmd += f' --model_name "{model_name}"'

    subprocess.run(cmd, shell=True, check=True)
    MODEL_VOLUME.commit()
    print(f"[protenix] Model downloaded to volume: {model}")


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
        cmd = f'stdbuf -oL protenix predict --input "{json_path}" --out_dir "{out_dir}" --seeds {seeds} --use_msa {use_msa_str}'
        if model == "protenix_mini":
            cmd += ' --model_name "protenix_mini_default_v0.5.0"'
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
