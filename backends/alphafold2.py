"""AlphaFold2/ColabFold structure prediction backend."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from modal import Image

from .common import (
    CACHE_VOLUME,
    GPU,
    LoggingOutput,
    MODEL_VOLUME,
    TIMEOUT,
    alphafold_cache_key,
    app,
)


# =============================================================================
# MSA Format Conversion
# =============================================================================


def _parse_a3m_blocks(content: str, num_chains: int) -> dict[int, str]:
    """Parse paired A3M into per-chain raw A3M blocks (including headers)."""
    blocks: dict[int, list[str]] = {i: [] for i in range(num_chains)}
    current_chain: int | None = None
    current_lines: list[str] = []

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):
            if current_chain is not None and current_lines:
                blocks[current_chain].extend(current_lines)
            current_lines = []
            header = line[1:].strip()
            first_field = header.split("\t")[0].strip()
            try:
                chain_id = int(first_field)
                if 101 <= chain_id < 101 + num_chains:
                    current_chain = chain_id - 101
            except ValueError:
                pass
            current_lines = [line]
        else:
            if current_chain is not None:
                current_lines.append(line)

    if current_chain is not None and current_lines:
        blocks[current_chain].extend(current_lines)

    return {k: "\n".join(v) + "\n" for k, v in blocks.items() if v}


def _clean_a3m(raw: str) -> str:
    """Normalize A3M for ColabFold: strip NUL/BOM, remove # comments and blank lines."""
    if not raw:
        return ""
    raw = raw.replace("\x00", "")
    raw = raw.lstrip("\ufeff")
    lines: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines) + ("\n" if lines else "")


def _read_a3m(path: Path) -> str:
    if not path.exists():
        return ""
    return _clean_a3m(path.read_text())


def _build_af2_queries(
    msa_result: dict,
    jobname: str,
    chain_sequences: list[str],
) -> tuple[list, bool]:
    """Build ColabFold queries list from pre-fetched MSAs.

    `chain_sequences` is the ordered list of protein chains in THIS input
    (homomer duplicates preserved). MSAs are looked up by sequence in
    `msa_result["unpaired"]` (which keys by unique sequence).

    Bypasses get_queries() to avoid parse_fasta() issues with serialized A3M.
    For monomers: raw A3M content (no header needed — unserialize_msa fallback).
    For multimers: uses msa_to_str() for proper #len\tcard header format.

    Returns (queries, is_complex).
    """
    unpaired = msa_result["unpaired"]
    paired_dir = msa_result.get("paired_dir")
    is_complex = len(chain_sequences) > 1

    unpaired_msa: list[str] = []
    for seq in chain_sequences:
        parts: list[str] = []
        if seq in unpaired:
            uniref_path = Path(unpaired[seq]) / "uniref.a3m"
            if uniref_path.exists():
                parts.append(uniref_path.read_text())
            bfd_path = Path(unpaired[seq]) / "bfd.mgnify30.metaeuk30.smag30.a3m"
            if bfd_path.exists():
                parts.append(bfd_path.read_text())
            if not parts:
                merged_path = Path(unpaired[seq]) / "merged.a3m"
                if merged_path.exists():
                    parts.append(merged_path.read_text())
        raw = "\n".join(parts) if parts else ""
        unpaired_msa.append(_clean_a3m(raw))

    if is_complex:
        from colabfold.input import msa_to_str

        paired_msa: list[str] | None = None
        if paired_dir:
            pair_a3m_path = Path(paired_dir) / "pair.a3m"
            if pair_a3m_path.exists():
                content = pair_a3m_path.read_text().replace("\x00", "")
                # ColabSearch built pair.a3m with one section per chain in the
                # full (duplicate-preserving) sequence list it was sent. Split
                # by the same chain count so homomers get all sections.
                blocks = _parse_a3m_blocks(content, len(chain_sequences))
                if any(blocks.values()):
                    paired_msa = [blocks.get(i, "") for i in range(len(chain_sequences))]

        a3m_str = msa_to_str(unpaired_msa, paired_msa, chain_sequences, [1] * len(chain_sequences))
        query_sequence = ":".join(chain_sequences)
    else:
        a3m_str = unpaired_msa[0]
        query_sequence = chain_sequences[0]

    n_seqs = sum(1 for line in a3m_str.splitlines() if line.startswith(">"))
    print(f"[alphafold2] Pre-fetched MSA: {n_seqs} entries, {is_complex=}, chains={len(chain_sequences)}")

    queries = [(jobname, query_sequence, [a3m_str], None)]
    return queries, is_complex


# =============================================================================
# Image
# =============================================================================

ALPHAFOLD2_VOLUME_DIR = "/models/alphafold2"
ALPHAFOLD2_MARKER = Path(ALPHAFOLD2_VOLUME_DIR) / "_DOWNLOADED"
COLABFOLD_GIT_REF = "a134f6a8f8de5c41c63cb874d07e1a334cb021bb"


def _download_alphafold2_models():
    """Download ColabFold/AF2 params to the model volume during image build."""
    from colabfold.download import download_alphafold_params

    cache_dir = Path(ALPHAFOLD2_VOLUME_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if ALPHAFOLD2_MARKER.exists():
        print("[alphafold2] Already on volume (marker present); skipping download")
        return

    for model_type in ("alphafold2_ptm", "alphafold2_multimer_v3"):
        print(f"[alphafold2] Downloading {model_type}...")
        download_alphafold_params(model_type, data_dir=cache_dir)

    ALPHAFOLD2_MARKER.touch()
    MODEL_VOLUME.commit()
    print("[alphafold2] Params downloaded")


alphafold_image = (
    Image.micromamba(python_version="3.11")
    .apt_install("wget", "git")
    .pip_install(
        f"colabfold[alphafold-minus-jax]@git+https://github.com/sokrypton/ColabFold@{COLABFOLD_GIT_REF}"
    )
    .micromamba_install("kalign2=2.04", "hhsuite=3.3.0", "pdbfixer=1.9", channels=["conda-forge", "bioconda"])
    .run_commands(
        'pip install --upgrade "jax[cuda12_pip]==0.5.3" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html',
        gpu="a10g",
    )
    .run_function(_download_alphafold2_models, volumes={"/models": MODEL_VOLUME}, timeout=3600)
)

# =============================================================================
# Prediction
# =============================================================================


@app.function(
    image=alphafold_image,
    timeout=TIMEOUT * 60,
    gpu="a100-80gb",  # A100-80GB for better JAX compatibility and more VRAM
    volumes={"/models": MODEL_VOLUME, "/cache": CACHE_VOLUME},
)
def alphafold_predict(params: dict[str, Any], overwrite: bool = False, job_id: str | None = None) -> list:
    """Run AlphaFold2/ColabFold structure prediction."""
    import sys
    import zipfile
    from io import BytesIO
    from tempfile import TemporaryDirectory
    from colabfold.batch import get_queries, run as colabfold_run

    MODEL_VOLUME.reload()
    af2_data_dir = Path(ALPHAFOLD2_VOLUME_DIR)
    if not ALPHAFOLD2_MARKER.exists():
        raise RuntimeError(
            f"AF2 params not on volume at {af2_data_dir}. "
            "Rebuild with: uv run modal run --force-build foldism.py"
        )

    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"  # Limit JAX memory to 90%
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["PYTHONUNBUFFERED"] = "1"

    # Set up logging early to capture all prints
    old_stdout, old_stderr = sys.stdout, sys.stderr
    logging_stdout = LoggingOutput(old_stdout, job_id, "alphafold2_logs")
    sys.stdout = sys.stderr = logging_stdout

    try:
        input_str = params["input_str"]
        input_name = params.get("input_name", "input.fasta")
        models = params.get("models", [1])
        num_recycles = params.get("num_recycles", 3)
        msa_result = params.get("msa_result")
        use_msa = params.get("use_msa", True)

        # Cache identity assumes use_msa=True ⇔ pre-fetched ColabSearch MSA.
        # Refuse use_msa=True without msa_result to prevent a direct caller from
        # poisoning the cache with colabfold's mmseqs2_uniref_env server output.
        if use_msa and not msa_result:
            raise RuntimeError(
                "[alphafold2] use_msa=True but no msa_result provided. "
                "Pre-fetch via colabsearch (foldism orchestrator does this) "
                "or pass use_msa=False to run msa_mode=single_sequence."
            )

        cache_key = alphafold_cache_key(params)
        cache_path = Path(f"/cache/alphafold2/{cache_key}")
        cache_marker = cache_path / "_COMPLETE"

        if not overwrite and cache_marker.exists():
            print(f"[alphafold2] Cache HIT: {cache_key} (returning cached result)")
            CACHE_VOLUME.reload()
            logging_stdout.flush()
            sys.stdout, sys.stderr = old_stdout, old_stderr
            return [(f.relative_to(cache_path), f.read_bytes()) for f in cache_path.glob("**/*") if f.is_file() and f.name != "_COMPLETE"]

        print(f"[alphafold2] Cache MISS: {cache_key}")

        with TemporaryDirectory() as in_dir, TemporaryDirectory() as out_dir:
            jobname = Path(input_name).stem

            if msa_result:
                # Parse the ORIGINAL multi-record FASTA (not the AF2-converted
                # form, which joins chains with ':' on a single line and would
                # be treated as one non-protein sequence by extract_chain_sequences).
                from backends.common import extract_chain_sequences
                original_fasta = params.get("original_fasta") or input_str
                chain_sequences = extract_chain_sequences(original_fasta)
                if not chain_sequences:
                    raise RuntimeError("[alphafold2] No protein chains found in input")
                queries, is_complex = _build_af2_queries(msa_result, jobname, chain_sequences)
            else:
                fasta_path = Path(in_dir) / input_name
                fasta_path.write_text(input_str)
                queries, is_complex = get_queries(in_dir)

            print(f"Running AF2 on {len(queries)} queries, {is_complex=}")

            # Explicit MSA mode: single_sequence when use_msa=False, else default
            # (colabfold's mmseqs2 path will be skipped since we supply queries).
            msa_mode = "single_sequence" if not use_msa else "mmseqs2_uniref_env"
            print(f"[alphafold2] msa_mode={msa_mode}, msa_result={'yes' if msa_result else 'no'}")

            colabfold_run(
                queries=queries,
                result_dir=out_dir,
                use_templates=False,
                num_recycles=num_recycles,
                num_models=len(models),
                model_order=models,
                num_relax=0,
                model_type="auto",
                is_complex=is_complex,
                data_dir=af2_data_dir,
                msa_mode=msa_mode,
                zip_results=False,
            )

            outputs = [
                (f.relative_to(out_dir), f.read_bytes())
                for f in Path(out_dir).rglob("*")
                if f.is_file()
            ]

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
