"""Boltz-2 structure prediction backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from modal import Image

from .common import (
    BOLTZ_BASE_PARAMS,
    CACHE_VOLUME,
    GPU,
    MODEL_VOLUME,
    TIMEOUT,
    _fasta_to_boltz_yaml,
    app,
    boltz_cache_key,
    job_store,
    write_log_line,
)


# =============================================================================
# MSA Format Conversion
# =============================================================================


def _parse_a3m_hits(content: str) -> list[str]:
    """Parse A3M into hit sequences, skipping query. Strips lowercase insertions."""
    seqs: list[str] = []
    current_lines: list[str] = []

    for line in content.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith(">"):
            if current_lines:
                seq = "".join(current_lines)
                seqs.append("".join(c for c in seq if not c.islower()))
            current_lines = []
        else:
            current_lines.append(line.strip())

    if current_lines:
        seq = "".join(current_lines)
        seqs.append("".join(c for c in seq if not c.islower()))

    return seqs[1:]  # skip query


def _parse_paired_hits(content: str, num_chains: int) -> dict[int, list[str]]:
    """Parse paired A3M into per-chain hit sequence lists, skipping query."""
    chains: dict[int, list[str]] = {i: [] for i in range(num_chains)}
    current_chain: int | None = None
    current_lines: list[str] = []

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):
            if current_chain is not None and current_lines:
                seq = "".join(current_lines)
                chains[current_chain].append("".join(c for c in seq if not c.islower()))
            current_lines = []
            first_field = line[1:].strip().split("\t")[0].strip()
            try:
                chain_id = int(first_field)
                if 101 <= chain_id < 101 + num_chains:
                    current_chain = chain_id - 101
            except ValueError:
                pass
        else:
            if current_chain is not None:
                current_lines.append(line)

    if current_chain is not None and current_lines:
        seq = "".join(current_lines)
        chains[current_chain].append("".join(c for c in seq if not c.islower()))

    # Skip first entry per chain (query)
    for idx in chains:
        if chains[idx]:
            chains[idx] = chains[idx][1:]

    return chains


def _a3m_to_boltz_csv(msa_result: dict, work_dir: str) -> dict[str, str]:
    """Convert pre-fetched A3Ms to Boltz CSV format with pairing keys.

    Boltz CSV format: key,sequence
      key=0: query sequence
      key=1,2,...: paired hits (matching key across chains = paired)
      key=-1: unpaired hits

    Returns dict mapping sequence -> CSV file path.
    """
    MAX_PAIRED = 8192
    MAX_TOTAL = 16384

    sequences = msa_result["sequences"]
    unpaired = msa_result["unpaired"]
    paired_dir = msa_result.get("paired_dir")

    paired_chains: dict[int, list[str]] | None = None
    if paired_dir:
        pair_path = Path(paired_dir) / "pair.a3m"
        if pair_path.exists():
            paired_chains = _parse_paired_hits(
                pair_path.read_text().replace("\x00", ""), len(sequences)
            )

    csv_paths: dict[str, str] = {}
    for chain_idx, seq in enumerate(sequences):
        rows = [f"0,{seq}"]

        # Paired hits
        n_paired = 0
        if paired_chains and chain_idx in paired_chains:
            for hit_idx, hit_seq in enumerate(paired_chains[chain_idx]):
                if not hit_seq.replace("-", ""):
                    continue
                if n_paired >= MAX_PAIRED:
                    break
                n_paired += 1
                rows.append(f"{hit_idx + 1},{hit_seq}")

        # Unpaired hits
        if seq in unpaired:
            merged_path = Path(unpaired[seq]) / "merged.a3m"
            if merged_path.exists():
                for hit_seq in _parse_a3m_hits(merged_path.read_text()):
                    if len(rows) >= MAX_TOTAL:
                        break
                    rows.append(f"-1,{hit_seq}")

        csv_path = Path(work_dir) / f"msa_chain_{chain_idx}.csv"
        csv_path.write_text("key,sequence\n" + "\n".join(rows) + "\n")
        csv_paths[seq] = str(csv_path)

        n_unpaired = len(rows) - 1 - n_paired
        print(f"[boltz2] MSA CSV chain {chain_idx}: {n_paired} paired + {n_unpaired} unpaired = {len(rows)} total")

    return csv_paths


# =============================================================================
# Image
# =============================================================================


def _download_boltz_models():
    """Download Boltz models and CCD mols directory to volume during image build."""
    from boltz.main import download_boltz1, download_boltz2
    from huggingface_hub import snapshot_download

    if not Path("/models/boltz1_conf.ckpt").exists():
        print("[boltz] Downloading boltz1 model...")
        download_boltz1(Path("/models"))
        print("[boltz] boltz1 model downloaded")

    if not Path("/models/boltz2_conf.ckpt").exists():
        print("[boltz] Downloading boltz2 model...")
        download_boltz2(Path("/models"))
        print("[boltz] boltz2 model downloaded")

    # Download CCD mols directory - required for Boltz to recognize chemical components
    import tarfile
    mols_dir = Path("/models/mols")
    if not mols_dir.exists() or not any(mols_dir.glob("*.cif")):
        print("[boltz] Downloading CCD mols.tar from HuggingFace...")
        # Download the mols.tar file from HuggingFace
        from huggingface_hub import hf_hub_download
        tar_path = hf_hub_download(
            repo_id="boltz-community/boltz-2",
            filename="mols.tar",
            cache_dir="/tmp",
        )
        print(f"[boltz] Extracting mols.tar to /models/mols...")
        mols_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path="/models")
        cif_count = len(list(mols_dir.glob("*.cif")))
        print(f"[boltz] CCD mols directory extracted ({cif_count} CIF files)")


boltz_image = (
    Image.micromamba()
    .apt_install("wget", "git", "gcc", "g++")
    .pip_install(
        "colabfold[alphafold-minus-jax]@git+https://github.com/sokrypton/ColabFold@acc0bf772f22feb7f887ad132b7313ff415c8a9f"
    )
    .micromamba_install("kalign2=2.04", "hhsuite=3.3.0", channels=["conda-forge", "bioconda"])
    .run_commands(
        'pip install --upgrade "jax[cuda12_pip]==0.5.3" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html',
        gpu="a10g",
    )
    .run_commands("python -m colabfold.download")
    .apt_install("build-essential")
    .pip_install("polars==1.19.0", "boltz==2.2.0", "huggingface_hub")
    .run_function(_download_boltz_models, gpu="a10g", volumes={"/models": MODEL_VOLUME})
)

# =============================================================================
# Prediction
# =============================================================================


@app.function(
    image=boltz_image,
    timeout=TIMEOUT * 60,
    gpu=GPU,
    volumes={"/models": MODEL_VOLUME, "/cache": CACHE_VOLUME},
)
def boltz2_predict(params: dict[str, Any], overwrite: bool = False, job_id: str | None = None) -> list:
    """Run Boltz-2 structure prediction."""
    import os
    import subprocess
    from tempfile import TemporaryDirectory

    # Set BOLTZ_CACHE to ensure all data (including CCD) is stored on volume
    os.environ["BOLTZ_CACHE"] = "/models"

    input_str = params["input_str"]
    use_msa = params.get("use_msa", True)
    msa_paths = params.get("msa_paths")
    msa_result = params.get("msa_result")
    original_fasta = params.get("original_fasta")
    params_str = params.get("params_str", BOLTZ_BASE_PARAMS)

    cache_key = boltz_cache_key(params)
    cache_path = Path(f"/cache/boltz2/{cache_key}")
    cache_marker = cache_path / "_COMPLETE"

    if not overwrite and cache_marker.exists():
        msg = f"[boltz2] Cache HIT: {cache_key} (returning cached result)"
        print(msg)
        write_log_line(job_id, "boltz_logs", msg, "boltz2")
        CACHE_VOLUME.reload()
        return [(f.relative_to(cache_path), f.read_bytes()) for f in cache_path.glob("**/*") if f.is_file() and f.name != "_COMPLETE"]

    msg = f"[boltz2] Cache MISS: {cache_key} (use_msa={use_msa})"
    print(msg)
    write_log_line(job_id, "boltz_logs", msg, "boltz2")

    with TemporaryDirectory() as in_dir, TemporaryDirectory() as out_dir:
        # Convert pre-fetched A3Ms to Boltz CSV format with paired keys
        if msa_result and original_fasta:
            csv_paths = _a3m_to_boltz_csv(msa_result, in_dir)
            input_str = _fasta_to_boltz_yaml(original_fasta, msa_paths=csv_paths)
            print(f"[boltz2] Using pre-fetched MSAs (CSV with pairing) for {len(csv_paths)} chains")
        elif msa_paths:
            if input_str.strip().startswith(">"):
                input_str = _fasta_to_boltz_yaml(input_str, msa_paths=msa_paths)
            # Don't add --use_msa_server since MSAs are pre-computed
        else:
            if input_str.strip().startswith(">"):
                input_str = _fasta_to_boltz_yaml(input_str)
            params_str = f"--use_msa_server {params_str}"

        input_path = Path(in_dir) / "input.yaml"
        input_path.write_text(input_str)

        cmd = f'stdbuf -oL boltz predict {input_path} --out_dir {out_dir} --cache /models {params_str}'
        print(f"Running: {cmd}")
        write_log_line(job_id, "boltz_logs", f"Command: {cmd}", "boltz2")

        # Use Popen to capture output in real-time with line-buffered output
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line-buffered (bufsize=0 doesn't work with text=True)
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        )

        # Stream output line by line with batched log writes
        import sys
        import time
        msa_stuck_start = None
        MSA_TIMEOUT = 180  # 3 minutes max for MSA fetching
        log_buffer = []
        last_flush = time.time()
        FLUSH_INTERVAL = 10.0  # Flush logs every 10 seconds

        def flush_logs():
            nonlocal log_buffer, last_flush
            if log_buffer and job_id:
                try:
                    existing = job_store.get(job_id, {})
                    logs = existing.get("boltz_logs", [])
                    logs.extend(log_buffer)
                    existing["boltz_logs"] = logs[-1000:]
                    job_store[job_id] = existing
                except Exception as e:
                    print(f"[boltz2] Failed to flush logs: {e}")
                log_buffer = []
            last_flush = time.time()

        while True:
            line = process.stdout.readline()
            if not line:  # EOF
                break
            line = line.rstrip()
            if line:  # Skip empty lines
                print(line, flush=True)
                log_buffer.append(line)

                # Flush logs periodically
                if time.time() - last_flush >= FLUSH_INTERVAL:
                    flush_logs()

                # Detect MSA server stuck - only PENDING and Sleeping indicate no progress
                is_msa_stuck = "PENDING" in line or "Sleeping" in line

                if is_msa_stuck:
                    if msa_stuck_start is None:
                        msa_stuck_start = time.time()
                    elif time.time() - msa_stuck_start > MSA_TIMEOUT:
                        process.kill()
                        raise RuntimeError(f"MSA server stuck for >{MSA_TIMEOUT}s - ColabFold may be throttling this IP")
                else:
                    msa_stuck_start = None  # Any other output = progress

        # Flush any remaining logs
        flush_logs()
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

        pred_dir = None
        for d in Path(out_dir).iterdir():
            if d.is_dir() and d.name.startswith("boltz_results"):
                pred_dir = d
                break

        if pred_dir is None:
            raise RuntimeError(f"No prediction directory found in {out_dir}")

        outputs = [(f.relative_to(pred_dir), f.read_bytes()) for f in pred_dir.rglob("*") if f.is_file()]

        cache_path.mkdir(parents=True, exist_ok=True)
        for rel_path, content in outputs:
            out_path = cache_path / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(content)

        cache_marker.touch()
        CACHE_VOLUME.commit()

        return outputs
