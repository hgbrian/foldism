"""OpenFold 3 structure prediction backend."""

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
    _fasta_to_openfold3_json,
    app,
    job_store,
    openfold3_cache_key,
    write_log_line,
)

# =============================================================================
# Model Configuration
# =============================================================================

OPENFOLD3_CHECKPOINT = "of3-p2-155k"

# =============================================================================
# MSA Format Conversion
# =============================================================================


def _split_paired_a3m(content: str, num_chains: int) -> list[str]:
    """Split combined paired A3M into per-chain A3M strings.

    The pair.a3m has chain sections marked by >101, >102, etc.
    Returns one A3M string per chain.
    """
    blocks: dict[int, list[str]] = {i: [] for i in range(num_chains)}
    current_chain: int | None = None

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(">"):
            first_field = stripped[1:].split("\t")[0].strip()
            try:
                chain_id = int(first_field)
                if 101 <= chain_id < 101 + num_chains:
                    current_chain = chain_id - 101
                    blocks[current_chain].append(">query")
                    continue
            except ValueError:
                pass
            if current_chain is not None:
                blocks[current_chain].append(stripped)
        else:
            if current_chain is not None:
                blocks[current_chain].append(stripped)

    return ["\n".join(blocks.get(i, [])) + "\n" for i in range(num_chains)]


# =============================================================================
# Image
# =============================================================================


def _download_openfold3_models():
    """Download OpenFold3 checkpoint and CCD to volume."""
    import subprocess

    cache_dir = Path("/models/openfold3")
    cache_dir.mkdir(parents=True, exist_ok=True)

    ckpt = cache_dir / f"{OPENFOLD3_CHECKPOINT}.pt"
    if not ckpt.exists():
        print(f"[openfold3] Downloading checkpoint {OPENFOLD3_CHECKPOINT}.pt...")
        subprocess.run([
            "aws", "s3", "cp", "--no-sign-request",
            f"s3://openfold3-data/openfold3-parameters/{OPENFOLD3_CHECKPOINT}.pt",
            str(ckpt),
        ], check=True)
        print(f"[openfold3] Checkpoint downloaded ({ckpt.stat().st_size:,} bytes)")

    (cache_dir / "ckpt_root").write_text(str(cache_dir))

    ccd = cache_dir / "components.bcif"
    if not ccd.exists():
        print("[openfold3] Downloading CCD components.bcif...")
        subprocess.run([
            "aws", "s3", "cp", "--no-sign-request",
            "s3://openfold3-data/components.bcif",
            str(ccd),
        ], check=True)
        print(f"[openfold3] CCD downloaded ({ccd.stat().st_size:,} bytes)")


openfold3_image = (
    Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "wget", "aria2", "libxrender1", "libxext6")
    .env({
        "OPENFOLD_CACHE": "/models/openfold3",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;9.0",
        "CUDA_HOME": "/usr/local/cuda",
    })
    .pip_install("openfold3==0.4.0")
    .run_function(_download_openfold3_models, volumes={"/models": MODEL_VOLUME})
)

# =============================================================================
# Prediction
# =============================================================================


@app.function(
    image=openfold3_image,
    timeout=TIMEOUT * 60,
    gpu="A100-40GB",
    volumes={"/models": MODEL_VOLUME, "/cache": CACHE_VOLUME},
)
def openfold3_predict(params: dict[str, Any], overwrite: bool = False, job_id: str | None = None) -> list:
    """Run OpenFold 3 structure prediction."""
    import subprocess
    from tempfile import TemporaryDirectory

    input_str = params["input_str"]
    input_name = params.get("input_name", "input")
    use_msa = params.get("use_msa", True)
    msa_result = params.get("msa_result")

    log_key = "openfold3_logs"

    MODEL_VOLUME.reload()
    CACHE_VOLUME.reload()
    ckpt_path = Path(f"/models/openfold3/{OPENFOLD3_CHECKPOINT}.pt")
    if not ckpt_path.exists():
        raise RuntimeError(
            f"Checkpoint not found: {ckpt_path}. "
            "Rebuild with: uv run modal run --force-build foldism.py"
        )
    write_log_line(job_id, log_key, "[openfold3] Models verified", "openfold3")

    cache_key = openfold3_cache_key(params)
    cache_path = Path(f"/cache/openfold3/{cache_key}")
    cache_marker = cache_path / "_COMPLETE"

    if not overwrite and cache_marker.exists():
        msg = f"[openfold3] Cache HIT: {cache_key} (returning cached result)"
        print(msg)
        write_log_line(job_id, log_key, msg, "openfold3")
        CACHE_VOLUME.reload()
        return [(f.relative_to(cache_path), f.read_bytes()) for f in cache_path.glob("**/*") if f.is_file() and f.name != "_COMPLETE"]

    msg = f"[openfold3] Cache MISS: {cache_key} (use_msa={use_msa})"
    print(msg)
    write_log_line(job_id, log_key, msg, "openfold3")

    with TemporaryDirectory() as in_dir, TemporaryDirectory() as out_dir:
        # Prepare MSA files in openfold3's expected format:
        #   main/<rep_id>/colabfold_main.a3m  (per-chain)
        #   paired/<complex_id>/<rep_id>/colabfold_paired.a3m  (per-chain)
        msa_dir_map: dict[str, str] = {}  # seq -> main msa dir
        paired_dir_map: dict[str, str] = {}  # seq -> paired msa dir
        if msa_result:
            msa_base = Path(in_dir) / "msas"
            sequences = msa_result.get("sequences", [])

            # Main (unpaired) MSAs — copy as colabfold_main.a3m per chain
            for chain_idx, seq in enumerate(sequences):
                unpaired_path = msa_result.get("unpaired", {}).get(seq)
                if unpaired_path:
                    merged = Path(unpaired_path) / "merged.a3m"
                    if merged.exists():
                        rep_id = f"chain_{chain_idx}"
                        chain_msa_dir = msa_base / "main" / rep_id
                        chain_msa_dir.mkdir(parents=True, exist_ok=True)
                        msa_content = merged.read_text()
                        (chain_msa_dir / "colabfold_main.a3m").write_text(msa_content)
                        msa_dir_map[seq] = str(chain_msa_dir)
                        n_seqs = msa_content.count("\n>")
                        print(f"[openfold3] Main MSA chain {chain_idx}: {merged.stat().st_size:,} bytes, {n_seqs} sequences")

            # Paired MSAs — split combined pair.a3m into per-chain colabfold_paired.a3m
            if msa_result.get("paired_dir"):
                pair_path = Path(msa_result["paired_dir"]) / "pair.a3m"
                if pair_path.exists():
                    per_chain = _split_paired_a3m(
                        pair_path.read_text().replace("\x00", ""),
                        len(sequences),
                    )
                    complex_id = "complex_0"
                    for chain_idx, (seq, a3m) in enumerate(zip(sequences, per_chain)):
                        # Skip empty or trivial paired MSAs
                        if not a3m.strip() or a3m.strip().count("\n") < 1:
                            print(f"[openfold3] Paired MSA chain {chain_idx}: empty, skipping")
                            continue
                        rep_id = f"chain_{chain_idx}"
                        paired_chain_dir = msa_base / "paired" / complex_id / rep_id
                        paired_chain_dir.mkdir(parents=True, exist_ok=True)
                        (paired_chain_dir / "colabfold_paired.a3m").write_text(a3m)
                        paired_dir_map[seq] = str(paired_chain_dir)
                        n_paired = a3m.count("\n>")
                        print(f"[openfold3] Paired MSA chain {chain_idx}: {len(a3m):,} bytes, {n_paired} sequences")
                    if paired_dir_map:
                        print(f"[openfold3] Split paired A3M into {len(paired_dir_map)} per-chain files")

        # Convert FASTA to openfold3 query JSON with MSA directory paths
        if input_str.strip().startswith(">"):
            input_str = _fasta_to_openfold3_json(
                input_str, input_name,
                msa_dir_map=msa_dir_map,
                paired_dir_map=paired_dir_map,
            )

        json_path = Path(in_dir) / "query.json"
        json_path.write_text(input_str)
        print(f"[openfold3] Query JSON:\n{input_str}")

        # Disable MSA server when pre-fetched MSAs are provided
        use_msa_server = not msa_result and use_msa

        # Write runner config to disable DeepSpeed evo attention (avoids JIT compilation issues)
        import yaml
        runner_config = {
            "model_update": {
                "presets": ["predict", "pae_enabled"],
                "custom": {
                    "settings": {
                        "memory": {
                            "eval": {
                                "use_deepspeed_evo_attention": False,
                            }
                        }
                    }
                },
            }
        }
        runner_yaml_path = Path(in_dir) / "runner.yaml"
        runner_yaml_path.write_text(yaml.dump(runner_config))

        cmd = (
            f'stdbuf -oL run_openfold predict'
            f' --query-json {json_path}'
            f' --output-dir {out_dir}'
            f' --inference-ckpt-path {ckpt_path}'
            f' --use-msa-server {str(use_msa_server).lower()}'
            f' --runner-yaml {runner_yaml_path}'
        )
        print(f"Running: {cmd}")
        write_log_line(job_id, log_key, f"Command: {cmd}", "openfold3")

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
                    print(f"[openfold3] Failed to flush logs: {e}")
                log_buffer = []
            last_flush = time.time()

        while True:
            line = process.stdout.readline()
            if not line:
                break
            line = line.rstrip()
            if line:
                print(line, flush=True)
                log_buffer.append(line)
                if time.time() - last_flush >= FLUSH_INTERVAL:
                    flush_logs()

        flush_logs()
        process.wait()
        if process.returncode != 0:
            # Print error log if available
            for log_file in Path(out_dir).rglob("*.log"):
                print(f"\n=== {log_file} ===")
                print(log_file.read_text()[-2000:])
            raise subprocess.CalledProcessError(process.returncode, cmd)

        # Collect all output files
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
