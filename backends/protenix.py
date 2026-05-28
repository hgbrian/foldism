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
# Model Configuration
# =============================================================================

PROTENIX_BASE_MODEL = "protenix-v2"
PROTENIX_HF_BUCKET = "btnaughton/bm9pc2VkcmF3"  # public HF bucket mirroring weights + CCD files
# HF "buckets" are versionless blob storage — unlike model repos they have no
# commit SHAs and `hf_hub_download(repo_type="model")` doesn't work for them.
# We download via plain HTTP from /buckets/<user>/<bucket>/resolve/<filename>.
# This string is a manual cache-buster: bump it whenever you re-upload files
# to the bucket so cached predictions don't silently reuse old weights.
PROTENIX_HF_REVISION = "2026-05-27"

# =============================================================================
# Image
# =============================================================================


def _setup_protenix_volume(prefix: str = "[protenix]") -> Path:
    """Symlink protenix's hardcoded data paths to the model volume.

    v1 looked at `<site-packages>/release_data`. v2 hardcodes `/root/common/`
    (for CCD files) and `/root/checkpoint/` (for the model `.pt`). We link
    all three so any version finds the pre-downloaded files on the volume
    instead of fetching from ByteDance at runtime/warmup.
    """
    import shutil
    import site

    volume_data = Path("/models/protenix_data")
    volume_data.mkdir(parents=True, exist_ok=True)
    (volume_data / "checkpoint").mkdir(parents=True, exist_ok=True)
    (volume_data / "ccd_cache").mkdir(parents=True, exist_ok=True)

    site_pkg = Path(site.getsitepackages()[0])
    links = [
        # (link_path, target)
        (site_pkg / "release_data", volume_data),                    # v1 legacy
        (Path("/root/checkpoint"), volume_data / "checkpoint"),      # v2 weights
        (Path("/root/common"), volume_data / "ccd_cache"),           # v2 CCD/cluster/obsolete
    ]
    for link, target in links:
        if link.is_symlink():
            if link.readlink() == target:
                continue
            link.unlink()
        elif link.exists():
            shutil.rmtree(link)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)
        print(f"{prefix} Symlinked {link} -> {target}")

    return volume_data


# Marker name includes the model version so bumping PROTENIX_BASE_MODEL
# automatically forces a fresh download instead of falsely short-circuiting
# on a marker left by an older model release.
PROTENIX_MARKER = Path(f"/models/protenix_data/_DOWNLOADED_{PROTENIX_BASE_MODEL}")


def _fetch_from_hf_bucket(filename: str, dest: Path, prefix: str = "[protenix]") -> bool:
    """Download `filename` from the public HF bucket to `dest` via plain HTTP.

    Buckets aren't accessible via huggingface_hub (it expects model/dataset/space
    repos); the public download URL is /buckets/<user>/<bucket>/resolve/<file>,
    which 302-redirects to a signed Cloudflare CAS URL. urlretrieve follows it.
    """
    if dest.exists() and dest.stat().st_size > 0:
        print(f"{prefix} Already exists: {dest.name} ({dest.stat().st_size:,} bytes)")
        return False

    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/buckets/{PROTENIX_HF_BUCKET}/resolve/{filename}"
    print(f"{prefix} Downloading {filename} from {url}...")
    urllib.request.urlretrieve(url, str(dest))
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"Download failed: {dest} (0 bytes)")
    print(f"{prefix} Downloaded: {dest.name} ({dest.stat().st_size:,} bytes)")
    return True


def _download_protenix_models():
    """Download protenix model + CCD files from HF bucket to VOLUME at image build,
    then always run the warmup so CUDA kernels are compiled into the image layer
    (even if the volume already had the files from a prior build).
    """
    expected_ckpt = Path("/models/protenix_data/checkpoint") / f"{PROTENIX_BASE_MODEL}.pt"
    if PROTENIX_MARKER.exists() and expected_ckpt.exists() and expected_ckpt.stat().st_size > 0:
        print(f"[BUILD] Protenix {PROTENIX_BASE_MODEL} files present on volume; skipping download")
    else:
        volume_data = _setup_protenix_volume(prefix="[BUILD]")
        checkpoint_dir = volume_data / "checkpoint"
        ccd_dir = volume_data / "ccd_cache"

        # (filename_in_bucket, destination_on_volume)
        downloads = [
            (f"{PROTENIX_BASE_MODEL}.pt", checkpoint_dir / f"{PROTENIX_BASE_MODEL}.pt"),
            ("components.cif", ccd_dir / "components.cif"),
            ("components.cif.rdkit_mol.pkl", ccd_dir / "components.cif.rdkit_mol.pkl"),
            ("clusters-by-entity-40.txt", ccd_dir / "clusters-by-entity-40.txt"),
            ("obsolete_release_date.csv", ccd_dir / "obsolete_release_date.csv"),
        ]
        for filename, dest in downloads:
            _fetch_from_hf_bucket(filename, dest, prefix="[BUILD]")
        PROTENIX_MARKER.parent.mkdir(parents=True, exist_ok=True)
        PROTENIX_MARKER.touch()
        MODEL_VOLUME.commit()

    # ALWAYS run the warmup so the JIT-compiled CUDA kernels (fast_layer_norm_*)
    # land in this image layer. Skipping this on rebuilds-with-files-already-present
    # forces every cold-start runtime to recompile, which is ~minutes per container.
    _setup_protenix_volume(prefix="[BUILD]")  # ensure symlink exists in this layer too
    import subprocess
    print("[BUILD] Warming up protenix (compiling CUDA kernels into image layer)...")
    test_json = '[{"name": "test", "sequences": [{"proteinChain": {"sequence": "MKTAYIAKQRQISFVKSH", "count": 1}}]}]'
    Path("/tmp/warmup").mkdir(parents=True, exist_ok=True)
    Path("/tmp/warmup/test.json").write_text(test_json)

    cmd = (
        f'protenix pred --input /tmp/warmup/test.json --out_dir /tmp/warmup_out '
        f'--seeds 42 --use_msa false --model_name "{PROTENIX_BASE_MODEL}" '
        f'--use_default_params true'
    )
    print(f"[BUILD] Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"[BUILD] stdout: {result.stdout}")
    print(f"[BUILD] stderr: {result.stderr}")
    if result.returncode != 0:
        raise RuntimeError(f"Warmup failed with code {result.returncode}: {result.stderr}")
    print("[BUILD] CUDA kernels compiled")


# =============================================================================
# MSA Format Conversion
# =============================================================================


def _split_paired_a3m(content: str, num_chains: int) -> list[str]:
    """Split combined paired A3M into per-chain A3M strings.

    The pair.a3m has chain sections marked by >101, >102, etc.
    Returns one A3M string per chain, with chain ID headers replaced by >query.
    Hit headers (with taxonomy info) are preserved for cross-chain pairing.
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


protenix_image = (
    Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "wget", "clang")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        # Cap ninja parallel compile workers so the build-time CUDA-kernel
        # warmup doesn't OOM-kill its workers and silently freeze at low %.
        "MAX_JOBS": "2",
    })
    .pip_install("protenix==2.0.0", "huggingface_hub>=0.27")
    .run_function(
        _download_protenix_models,
        gpu="L40S",
        volumes={"/models": MODEL_VOLUME},
        timeout=3600,
    )
)

# =============================================================================
# Model Download
# =============================================================================


def _ensure_protenix_models():
    """Verify Protenix models exist on volume (should be pre-downloaded at image build)."""
    MODEL_VOLUME.reload()
    volume_data = _setup_protenix_volume()

    checkpoint_file = volume_data / "checkpoint" / f"{PROTENIX_BASE_MODEL}.pt"

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
    msa_result = params.get("msa_result")

    log_key = "protenix_logs"

    # Check models exist (read-only to allow parallel execution)
    write_log_line(job_id, log_key, "[protenix] Checking models...", "protenix")
    _ensure_protenix_models()

    cache_key = protenix_cache_key(params)
    cache_path = Path(f"/cache/protenix/{cache_key}")
    cache_marker = cache_path / "_COMPLETE"

    if not overwrite and cache_marker.exists():
        msg = f"[protenix] Cache HIT: {cache_key} (returning cached result)"
        print(msg)
        write_log_line(job_id, log_key, msg, "protenix")
        CACHE_VOLUME.reload()
        return [(f.relative_to(cache_path), f.read_bytes()) for f in cache_path.glob("**/*") if f.is_file() and f.name != "_COMPLETE"]

    msg = f"[protenix] Cache MISS: {cache_key} (use_msa={use_msa})"
    print(msg)
    write_log_line(job_id, log_key, msg, "protenix")

    with TemporaryDirectory() as in_dir, TemporaryDirectory() as out_dir:
        # Split the combined pair.a3m by FULL chain count (duplicates preserved),
        # so homomer chains each get their own paired MSA section.
        if msa_result and msa_result.get("paired_dir"):
            from .common import extract_chain_sequences
            chain_sequences = extract_chain_sequences(input_str)
            pair_path = Path(msa_result["paired_dir"]) / "pair.a3m"
            if pair_path.exists() and len(chain_sequences) > 1:
                per_chain = _split_paired_a3m(
                    pair_path.read_text().replace("\x00", ""),
                    len(chain_sequences),
                )
                msa_result = dict(msa_result)  # shallow copy
                paired_per_chain_list: list[str | None] = []
                for chain_idx, a3m in enumerate(per_chain):
                    chain_file = Path(in_dir) / f"paired_chain_{chain_idx}.a3m"
                    chain_file.write_text(a3m)
                    paired_per_chain_list.append(str(chain_file))
                msa_result["paired_per_chain"] = paired_per_chain_list
                print(f"[protenix] Split paired A3M into {len(paired_per_chain_list)} per-chain files")

        if input_str.strip().startswith(">"):
            input_str = _fasta_to_protenix_json(input_str, input_name, msa_result=msa_result)

        json_path = Path(in_dir) / "input.json"
        json_path.write_text(input_str)

        # Never let protenix call its own MSA server (protenix-server.com, bytedance-hosted).
        # Callers must pre-fetch via colabsearch — guarded above. When we DO have
        # msa_result, we must pass --use_msa true so protenix reads our pre-fetched
        # unpairedMsaPath/pairedMsaPath entries from the JSON (under --use_msa false
        # protenix silently ignores them and runs single-sequence).
        if not msa_result and use_msa:
            raise RuntimeError(
                "[protenix] use_msa=True but no msa_result was provided. "
                "Refusing to fall back to protenix's built-in MSA server. "
                "Either supply pre-fetched MSAs (via foldism's colabsearch path) "
                "or pass use_msa=False to run single-sequence."
            )
        os.environ["MMSEQS_SERVICE_HOST_URL"] = "DISABLED"  # belt-and-suspenders
        use_msa_str = "true" if msa_result else "false"
        cmd = (
            f'stdbuf -oL protenix pred --input "{json_path}" --out_dir "{out_dir}" '
            f'--seeds {seeds} --use_msa {use_msa_str} --model_name "{PROTENIX_BASE_MODEL}" '
            f'--use_default_params true'
        )
        print(f"Running: {cmd}")
        write_log_line(job_id, log_key, f"Command: {cmd}", "protenix")

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
                    print(f"[protenix] Failed to flush logs: {e}")
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
