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
# MSA Format Conversion
# =============================================================================


def _parse_a3m_with_headers(content: str) -> list[tuple[str, str]]:
    """Parse A3M file into (header, sequence) tuples. Skips query. Strips lowercase."""
    results: list[tuple[str, str]] = []
    current_header = ""
    current_seq_lines: list[str] = []
    is_first = True

    for line in content.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith(">"):
            if current_seq_lines:
                seq = "".join(current_seq_lines)
                seq = "".join(c for c in seq if not c.islower())
                if is_first:
                    is_first = False
                else:
                    results.append((current_header, seq))
            current_header = line[1:].strip()
            current_seq_lines = []
        else:
            current_seq_lines.append(line.strip())

    if current_seq_lines:
        seq = "".join(current_seq_lines)
        seq = "".join(c for c in seq if not c.islower())
        if not is_first:
            results.append((current_header, seq))

    return results


def _parse_paired_a3m_chai(content: str, num_chains: int) -> dict[int, list[tuple[str, str]]]:
    """Parse paired A3M into per-chain (header, sequence) lists. Skips query."""
    chain_seqs: dict[int, list[tuple[str, str]]] = {i: [] for i in range(num_chains)}
    current_chain: int | None = None
    current_header = ""
    current_seq_lines: list[str] = []

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):
            if current_chain is not None and current_seq_lines:
                chain_idx = current_chain - 101
                if 0 <= chain_idx < num_chains:
                    seq = "".join(current_seq_lines)
                    seq = "".join(c for c in seq if not c.islower())
                    chain_seqs[chain_idx].append((current_header, seq))
            current_seq_lines = []

            header = line[1:].strip()
            current_header = header
            # Chain IDs are standalone headers like ">101" or ">102\tinfo"
            # Do NOT scan all tab fields — alignment positions can be 101/102
            first_field = header.split("\t")[0].strip()
            try:
                chain_id = int(first_field)
                if 101 <= chain_id < 101 + num_chains:
                    current_chain = chain_id
            except ValueError:
                pass  # keep current_chain — this is a hit within the current chain
        else:
            if current_chain is not None:
                current_seq_lines.append(line)

    if current_chain is not None and current_seq_lines:
        chain_idx = current_chain - 101
        if 0 <= chain_idx < num_chains:
            seq = "".join(current_seq_lines)
            seq = "".join(c for c in seq if not c.islower())
            chain_seqs[chain_idx].append((current_header, seq))

    # Skip query sequences (first in each chain)
    for idx in chain_seqs:
        if chain_seqs[idx]:
            chain_seqs[idx] = chain_seqs[idx][1:]

    return chain_seqs


def _a3m_to_chai_parquet(msa_result: dict, work_dir: Path) -> Path:
    """Convert A3Ms from colabsearch to Chai-1 Parquet format.

    Produces one Parquet per chain: {sha256(seq.upper())}.aligned.pqt
    Returns path to directory containing the parquet files.
    """
    import polars as pl
    from hashlib import sha256

    sequences = msa_result["sequences"]
    unpaired = msa_result["unpaired"]
    paired_dir = msa_result.get("paired_dir")

    paired_hits: dict[int, list[tuple[str, str]]] | None = None
    if paired_dir:
        pair_a3m_path = Path(paired_dir) / "pair.a3m"
        if pair_a3m_path.exists():
            paired_hits = _parse_paired_a3m_chai(pair_a3m_path.read_text().replace("\x00", ""), len(sequences))

    msa_dir = work_dir / "msas"
    msa_dir.mkdir(parents=True, exist_ok=True)

    for chain_idx, seq in enumerate(sequences):
        rows: list[dict[str, str]] = []

        # Query row
        rows.append({"sequence": seq.upper(), "source_database": "query", "pairing_key": "", "comment": ""})

        # Paired rows
        if paired_hits and chain_idx in paired_hits:
            for row_idx, (header, hit_seq) in enumerate(paired_hits[chain_idx]):
                if hit_seq.replace("-", ""):
                    rows.append({"sequence": hit_seq, "source_database": "uniref90", "pairing_key": str(row_idx), "comment": header})

        # Unpaired from uniref
        if seq in unpaired:
            uniref_path = Path(unpaired[seq]) / "uniref.a3m"
            if uniref_path.exists():
                for header, hit_seq in _parse_a3m_with_headers(uniref_path.read_text()):
                    rows.append({"sequence": hit_seq, "source_database": "uniref90", "pairing_key": "", "comment": header})

            # Unpaired from bfd/env databases
            bfd_path = Path(unpaired[seq]) / "bfd.mgnify30.metaeuk30.smag30.a3m"
            if bfd_path.exists():
                for header, hit_seq in _parse_a3m_with_headers(bfd_path.read_text()):
                    rows.append({"sequence": hit_seq, "source_database": "bfd_uniclust", "pairing_key": "", "comment": header})

        df = pl.DataFrame(rows)
        seq_hash = sha256(seq.upper().encode()).hexdigest()
        pqt_path = msa_dir / f"{seq_hash}.aligned.pqt"
        df.write_parquet(pqt_path)
        print(f"[chai1] MSA Parquet chain {chain_idx}: {len(rows)} rows -> {pqt_path.name}")

    return msa_dir

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
        msa_result = params.get("msa_result")

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

            # Convert pre-computed MSAs to Chai parquet format
            msa_directory = None
            if msa_result:
                msa_directory = _a3m_to_chai_parquet(msa_result, Path(in_dir))
                use_msa_server = False

            run_inference(
                fasta_file=fasta_path,
                output_dir=Path(out_dir),
                msa_directory=msa_directory,
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
