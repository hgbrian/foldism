"""ESMFold2 (Biohub) structure prediction backend.

Single-sequence (or multi-chain complex) folding via the open-source
`esm` package at https://github.com/Biohub/esm — no API token required.
Model weights are pulled from HuggingFace (biohub/ESMFold2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from modal import Image

from .common import (
    CACHE_VOLUME,
    MODEL_VOLUME,
    TIMEOUT,
    app,
    esmfold2_cache_key,
    fasta_iter,
    write_log_line,
)

# =============================================================================
# Model Configuration
# =============================================================================

ESMFOLD2_HF_REPO = "biohub/ESMFold2"
# Pinned to the HF commit SHA we built/tested against. To upgrade: pick a new
# SHA from huggingface.co/biohub/ESMFold2/commits/main, bump this constant
# (which also bumps the cache key via esmfold2_cache_key), and rebuild.
ESMFOLD2_HF_REVISION = "1afea82e432079d9af2ebd71d1e4c339ecca2ff0"
ESMFOLD2_VOLUME_DIR = "/models/esmfold2"
ESMFOLD2_GIT_REF = "c94ed8d"

# =============================================================================
# Image
# =============================================================================


ESMFOLD2_MARKER = Path(ESMFOLD2_VOLUME_DIR) / "_DOWNLOADED"


def _download_esmfold2_models():
    """Pre-download ESMFold2 weights (including LFS shards) to the model volume."""
    from huggingface_hub import snapshot_download

    cache_dir = Path(ESMFOLD2_VOLUME_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if ESMFOLD2_MARKER.exists():
        print(f"[BUILD] ESMFold2 already on volume (marker present); skipping download")
        return

    print(f"[BUILD] Downloading {ESMFOLD2_HF_REPO}@{ESMFOLD2_HF_REVISION} to {cache_dir}...")
    snapshot_download(
        repo_id=ESMFOLD2_HF_REPO,
        revision=ESMFOLD2_HF_REVISION,
        cache_dir=str(cache_dir),
        allow_patterns=["*.safetensors", "*.bin", "*.json", "*.pkl", "*.txt", "*.model"],
    )
    ESMFOLD2_MARKER.touch()
    MODEL_VOLUME.commit()
    print(f"[BUILD] ESMFold2 weights downloaded ({sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file()):,} bytes)")


esmfold2_image = (
    Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "wget")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "HF_HOME": ESMFOLD2_VOLUME_DIR,
        "HF_HUB_CACHE": ESMFOLD2_VOLUME_DIR,
    })
    .pip_install(
        f"esm @ git+https://github.com/Biohub/esm.git@{ESMFOLD2_GIT_REF}",
        "xformers",
    )
    .run_function(
        _download_esmfold2_models,
        volumes={"/models": MODEL_VOLUME},
        timeout=3600,
    )
)


# =============================================================================
# FASTA → ESMFold2 input
# =============================================================================


def _fasta_to_esmfold2_input(fasta_str: str, msa_result: dict | None = None):
    """Build a StructurePredictionInput from a FASTA string.

    Header type tags (`protein|`, `dna|`, `rna|`, `ligand|`) are honored when
    present; otherwise sequences are treated as protein. Ligand sequences are
    interpreted as SMILES. If `msa_result` is provided, each protein chain is
    given its unpaired MSA (loaded from colabsearch's merged.a3m).
    """
    from esm.models.esmfold2 import (
        DNAInput,
        LigandInput,
        ProteinInput,
        StructurePredictionInput,
    )

    try:
        from esm.models.esmfold2 import RNAInput
    except ImportError:
        RNAInput = None

    msa_for_seq: dict = {}
    if msa_result and msa_result.get("unpaired"):
        from esm.utils.msa.msa import MSA

        for seq, dir_path in msa_result["unpaired"].items():
            merged = Path(dir_path) / "merged.a3m"
            if merged.exists():
                msa_for_seq[seq] = MSA.from_a3m(str(merged))
        print(f"[esmfold2] Loaded MSAs for {len(msa_for_seq)} unique protein sequences")

    chain_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    sequences: list = []
    for n, (seq_id, seq) in enumerate(fasta_iter(fasta_str)):
        first = seq_id.split("|")[0].lower() if "|" in seq_id else ""
        cid = chain_letters[n] if n < len(chain_letters) else f"chain_{n}"
        if first == "ligand":
            sequences.append(LigandInput(id=cid, smiles=seq))
        elif first == "dna":
            sequences.append(DNAInput(id=cid, sequence=seq))
        elif first == "rna":
            if RNAInput is None:
                raise ValueError("RNA inputs not supported by this esm version")
            sequences.append(RNAInput(id=cid, sequence=seq))
        else:
            sequences.append(ProteinInput(id=cid, sequence=seq, msa=msa_for_seq.get(seq)))

    return StructurePredictionInput(sequences=sequences)


# =============================================================================
# Prediction
# =============================================================================


@app.function(
    image=esmfold2_image,
    timeout=TIMEOUT * 60,
    gpu="A100-40GB",
    volumes={"/models": MODEL_VOLUME, "/cache": CACHE_VOLUME},
)
def esmfold2_predict(params: dict[str, Any], overwrite: bool = False, job_id: str | None = None) -> list:
    """Run ESMFold2 structure prediction."""
    import json
    import time

    input_str = params["input_str"]
    input_name = params.get("input_name", "input")
    seed = params.get("seed", 42)
    num_samples = params.get("num_diffusion_samples", 1)
    num_steps = params.get("num_sampling_steps", 50)  # Biohub README example value
    num_loops = params.get("num_loops", 3)
    use_msa = params.get("use_msa", True)
    msa_result = params.get("msa_result") if use_msa else None

    # Cache key includes use_msa, so True must mean "MSA actually attached".
    if use_msa and not msa_result:
        raise RuntimeError(
            "[esmfold2] use_msa=True but no msa_result provided. "
            "Pre-fetch via colabsearch or pass use_msa=False for PLM-only."
        )

    log_key = "esmfold2_logs"
    write_log_line(job_id, log_key, "[esmfold2] Starting...", "esmfold2")

    cache_key = esmfold2_cache_key(params)
    cache_path = Path(f"/cache/esmfold2/{cache_key}")
    cache_marker = cache_path / "_COMPLETE"

    if not overwrite and cache_marker.exists():
        msg = f"[esmfold2] Cache HIT: {cache_key}"
        print(msg)
        write_log_line(job_id, log_key, msg, "esmfold2")
        CACHE_VOLUME.reload()
        return [(f.relative_to(cache_path), f.read_bytes()) for f in cache_path.glob("**/*") if f.is_file() and f.name != "_COMPLETE"]

    msg = f"[esmfold2] Cache MISS: {cache_key}"
    print(msg)
    write_log_line(job_id, log_key, msg, "esmfold2")

    from esm.models.esmfold2 import ESMFold2InputBuilder
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    MODEL_VOLUME.reload()
    t0 = time.time()
    model = ESMFold2Model.from_pretrained(
        ESMFOLD2_HF_REPO,
        revision=ESMFOLD2_HF_REVISION,
        local_files_only=True,  # rely on volume cache populated at build time
    ).cuda().eval()
    msg = f"[esmfold2] Model loaded in {time.time() - t0:.1f}s"
    print(msg)
    write_log_line(job_id, log_key, msg, "esmfold2")

    spi = _fasta_to_esmfold2_input(input_str, msa_result=msa_result)
    n_entities = len(spi.sequences)
    write_log_line(job_id, log_key, f"[esmfold2] {n_entities} entities, samples={num_samples}, steps={num_steps}, use_msa={use_msa and msa_result is not None}", "esmfold2")

    t0 = time.time()
    result = ESMFold2InputBuilder().fold(
        model, spi,
        num_loops=num_loops,
        num_sampling_steps=num_steps,
        num_diffusion_samples=num_samples,
        seed=seed,
        complex_id=input_name,
    )
    msg = f"[esmfold2] fold() done in {time.time() - t0:.1f}s"
    print(msg)
    write_log_line(job_id, log_key, msg, "esmfold2")

    results = result if isinstance(result, list) else [result]

    def _to_python(v):
        """Convert torch tensors / numpy arrays / scalars / containers to JSON-serializable Python."""
        if v is None:
            return None
        if hasattr(v, "tolist"):
            return v.tolist()
        if isinstance(v, dict):
            return {str(k): _to_python(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [_to_python(x) for x in v]
        return v

    outputs: list[tuple[Path, bytes]] = []
    for idx, sample in enumerate(results):
        complex_obj = getattr(sample, "complex", sample)
        cif_str = complex_obj.to_mmcif()
        plddt = complex_obj.plddt
        plddt_list = plddt.tolist() if hasattr(plddt, "tolist") else list(plddt)
        mean_plddt = (sum(plddt_list) / len(plddt_list)) if plddt_list else 0.0
        # ESMFold 2's MolecularComplexResult also exposes ptm / iptm / pair_chains_iptm
        # at the result (not complex) level — pull them when present so the UI
        # can show interface confidence the same way it does for the AF3-style backends.
        scores = {
            "plddt": mean_plddt,
            "plddt_per_token": plddt_list,
            "num_tokens": len(plddt_list),
            "sample_index": idx,
            "ptm": _to_python(getattr(sample, "ptm", None)),
            "iptm": _to_python(getattr(sample, "iptm", None)),
            "chain_pair_iptm": _to_python(getattr(sample, "pair_chains_iptm", None)),
        }
        outputs.append((Path(f"{input_name}_sample_{idx}.cif"), cif_str.encode()))
        outputs.append((Path(f"{input_name}_sample_{idx}_scores.json"), json.dumps(scores).encode()))

    cache_path.mkdir(parents=True, exist_ok=True)
    for rel_path, content in outputs:
        out_path = cache_path / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(content)
    cache_marker.touch()
    CACHE_VOLUME.commit()

    return outputs
