"""Common utilities and shared resources for all backends."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from hashlib import sha256
from io import StringIO
from itertools import groupby
from typing import Any

from modal import App, Dict, Image, Volume

# =============================================================================
# App and Configuration
# =============================================================================

app = App("foldism")
job_store = Dict.from_name("foldism-jobs", create_if_missing=True)

# GPU configuration - L40S default, configurable via environment
GPU = os.environ.get("GPU", "L40S")
TIMEOUT = int(os.environ.get("TIMEOUT", 60))  # 60 minutes default
CACHE_VERSION = "v2"  # v2: per-chain paired MSA, AF2 single_sequence mode, unified.json schema

# =============================================================================
# Real-time Log Streaming
# =============================================================================


def write_log_line(job_id: str | None, log_key: str, line: str, method_name: str = "backend"):
    """Write a single log line to job_store if job_id provided."""
    if job_id:
        try:
            existing = job_store.get(job_id, {})
            logs = existing.get(log_key, [])
            logs.append(line)
            existing[log_key] = logs[-1000:]  # Keep last 1000 lines
            job_store[job_id] = existing
        except Exception as e:
            print(f"[{method_name}] Failed to write log: {e}")


class LoggingOutput:
    """File-like object that writes to both stdout and job_store with batched writes."""

    def __init__(self, original_stdout, job_id: str | None, log_key: str):
        self.original_stdout = original_stdout
        self.job_id = job_id
        self.log_key = log_key
        self.buffer = ""
        self.log_buffer = []
        self.last_flush = __import__("time").time()
        self.FLUSH_INTERVAL = 10.0  # Flush every 10 seconds

    def _flush_logs(self):
        """Flush accumulated logs to job_store."""
        import time
        if self.log_buffer and self.job_id:
            try:
                existing = job_store.get(self.job_id, {})
                logs = existing.get(self.log_key, [])
                logs.extend(self.log_buffer)
                existing[self.log_key] = logs[-1000:]
                job_store[self.job_id] = existing
            except Exception as e:
                self.original_stdout.write(f"[{self.log_key}] Failed to flush logs: {e}\n")
            self.log_buffer = []
        self.last_flush = time.time()

    def write(self, text):
        import re
        import time
        self.original_stdout.write(text)
        self.original_stdout.flush()

        # Buffer text and process line by line (handle both \n and \r for tqdm)
        self.buffer += text
        while re.search(r'[\n\r]', self.buffer):
            match = re.search(r'[\n\r]', self.buffer)
            line = self.buffer[:match.start()]
            self.buffer = self.buffer[match.end():]
            if line.strip():
                self.log_buffer.append(line.strip())

        # Flush logs periodically
        if time.time() - self.last_flush >= self.FLUSH_INTERVAL:
            self._flush_logs()

    def flush(self):
        self.original_stdout.flush()
        self._flush_logs()  # Also flush logs on explicit flush

# =============================================================================
# Volumes (2 total: models + cache)
# =============================================================================

MODEL_VOLUME = Volume.from_name("foldism-models", create_if_missing=True)
CACHE_VOLUME = Volume.from_name("foldism-cache", create_if_missing=True)

# =============================================================================
# Folding App Definitions
# =============================================================================


@dataclass
class FoldingApp:
    name: str
    key: str
    description: str
    input_format: str


FOLDING_APPS: dict[str, FoldingApp] = {
    "opendde": FoldingApp("OpenDDE", "opendde", "OpenDDE v1", "normalized_fasta"),
    "mol1": FoldingApp("Mol-1", "mol1", "Mol-1 base", "normalized_fasta"),
    "alphafold2": FoldingApp("AlphaFold2", "alphafold2", "AlphaFold2/ColabFold", "af2_fasta"),
    "esmfold2": FoldingApp("ESMFold 2", "esmfold2", "ESMFold 2 (Biohub, single-sequence)", "esmfold2_fasta"),
}

# =============================================================================
# FASTA Utilities
# =============================================================================

ALLOWED_AAS = set("ACDEFGHIKLMNPQRSTVWYX")


def fasta_iter(s: str):
    """Iterate over FASTA entries yielding (header, sequence) tuples."""
    with StringIO(s) as fh:
        faiter = (x[1] for x in groupby(fh, lambda line: line.startswith(">")))
        for header in faiter:
            header = next(header)[1:].strip()
            seq = "".join(s.strip() for s in next(faiter))
            yield header, seq


def extract_chain_sequences(fasta_str: str) -> list[str]:
    """Return the ordered list of protein chain sequences in the FASTA.

    Preserves order AND duplicates (so a homodimer of [X, X] returns [X, X],
    not [X]). Skips non-protein entries (dna|, rna|, ligand|, ion|) and any
    sequence that isn't valid amino acids.

    This is the single source of truth for "what chains am I folding?" inside
    every backend. Never iterate `msa_result["sequences"]` for that purpose —
    it is deduplicated and will collapse homomers.
    """
    chains: list[str] = []
    for seq_id, seq in fasta_iter(fasta_str):
        first_part = seq_id.split("|")[0].lower() if "|" in seq_id else ""
        if first_part in {"dna", "rna", "ligand", "ion"}:
            continue
        if all(aa.upper() in ALLOWED_AAS for aa in seq):
            chains.append(seq)
    return chains


# =============================================================================
# Format Converters
# =============================================================================


def _fasta_to_af2_fasta(fasta_str: str) -> str:
    """Convert FASTA to AlphaFold2/ColabFold format.

    Output format: ><hash>
    SEQ1:SEQ2:SEQ3  (chains joined with colons on single line)
    """
    non_protein_types = {"dna", "rna", "ligand", "ion"}
    seqs = []
    for seq_id, seq in fasta_iter(fasta_str):
        first_part = seq_id.split("|")[0].lower() if "|" in seq_id else ""
        if first_part not in non_protein_types:
            seqs.append(seq)
    if not seqs:
        return fasta_str
    seq_hash = sha256(":".join(seqs).encode()).hexdigest()[:6]
    return f">{seq_hash}\n{':'.join(seqs)}"


def _normalize_fasta(fasta_str: str) -> str:
    """Normalize FASTA to use deterministic headers based on sequence hash."""
    seqs = [(seq_id, seq) for seq_id, seq in fasta_iter(fasta_str)]
    seq_hash = sha256(":".join(seq for _, seq in seqs).encode()).hexdigest()[:6]
    lines = []
    for i, (_, seq) in enumerate(seqs):
        lines.append(f">{seq_hash}_{i}")
        lines.append(seq)
    return "\n".join(lines)


CONVERTERS = {
    "normalized_fasta": _normalize_fasta,
    "af2_fasta": _fasta_to_af2_fasta,
    "esmfold2_fasta": _normalize_fasta,
}


def convert_for_app(fasta_str: str, app_key: str) -> str:
    app_def = FOLDING_APPS[app_key]
    return CONVERTERS[app_def.input_format](fasta_str)


# =============================================================================
# Cache Key Functions
# =============================================================================

def alphafold_cache_key(params: dict[str, Any]) -> str:
    """Generate cache key for AlphaFold2 predictions."""
    from .alphafold2 import COLABFOLD_GIT_REF
    input_hash = sha256(params.get("input_str", "").encode()).hexdigest()
    # `use_msa=True` means a pre-fetched colabsearch MSA is supplied;
    # `use_msa=False` lets colabfold use single_sequence mode.
    # These produce different outputs and must cache separately.
    cache_params = {
        "version": CACHE_VERSION,
        "scores_version": "protein-pairs-v1-ipsae-20260103",
        "calculate_actifptm": bool(params.get("calculate_actifptm", True)),
        "input_hash": input_hash,
        "models": params.get("models", [1]),
        "num_recycles": params.get("num_recycles", 3),
        "use_msa": params.get("use_msa", True),
        "colabfold_rev": COLABFOLD_GIT_REF,
    }
    if params.get("download_confidence_data", True):
        cache_params["download_confidence_data"] = True
    return sha256(json.dumps(cache_params, sort_keys=True).encode()).hexdigest()[:16]


def esmfold2_cache_key(params: dict[str, Any]) -> str:
    """Generate cache key for ESMFold2 predictions."""
    from .esmfold2 import ESMC_HF_REPO, ESMC_HF_REVISION, ESMFOLD2_HF_REPO, ESMFOLD2_HF_REVISION
    input_hash = sha256(params.get("input_str", "").encode()).hexdigest()
    cache_params = {
        "version": CACHE_VERSION,
        "scores_version": "protein-pairs-v1-ipsae-20260103",
        "calculate_actifptm": bool(params.get("calculate_actifptm", True)),
        "input_hash": input_hash,
        "seed": params.get("seed", 42),
        "num_diffusion_samples": params.get("num_diffusion_samples", 1),
        "num_sampling_steps": params.get("num_sampling_steps", 50),
        "num_loops": params.get("num_loops", 3),
        "use_msa": params.get("use_msa", True),
        "model": "esmfold2",
        "checkpoint": f"{ESMFOLD2_HF_REPO}@{ESMFOLD2_HF_REVISION}",
        "encoder_checkpoint": f"{ESMC_HF_REPO}@{ESMC_HF_REVISION}",
        # Bumped when scores.json schema grows (added ptm/iptm/chain_pair_iptm).
        # Old cached entries had only plddt fields and would mask the new metrics.
        "scores_v": "v2",
    }
    if params.get("download_confidence_data", True):
        cache_params["download_confidence_data"] = True
    return sha256(json.dumps(cache_params, sort_keys=True).encode()).hexdigest()[:16]



def native_confidence_cache_key(model: str, revision: str, params: dict) -> str:
    """Separate model, inference settings, and confidence capabilities."""
    values = {"model": model, "revision": revision, "scores_version": "protein-pairs-v1-ipsae-20260103",
              "input": params.get("input_str", ""), "seed": params.get("seed", 42),
              "num_samples": params.get("num_samples", 1), "num_steps": params.get("num_steps", 200),
              "num_cycles": params.get("num_cycles", 20 if model == "mol1" else 10),
              "use_msa": params.get("use_msa", True),
              "calculate_actifptm": bool(params.get("calculate_actifptm", True))}
    if params.get("download_confidence_data", True):
        values["download_confidence_data"] = True
    return sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:16]


def opendde_cache_key(params: dict) -> str:
    return native_confidence_cache_key("opendde", "eddd563ce96571f784012edd8f045181c8f8627d", params)


def mol1_cache_key(params: dict) -> str:
    return native_confidence_cache_key("mol1", "e6e64866ee89640a91cef7ee777a85f1d8be03bb", params)


def get_cache_key(method: str, params: dict[str, Any]) -> str:
    """Get cache key for any method."""
    if method == "opendde":
        return opendde_cache_key(params)
    if method == "mol1":
        return mol1_cache_key(params)
    if method == "alphafold2":
        return alphafold_cache_key(params)
    elif method == "esmfold2":
        return esmfold2_cache_key(params)
    else:
        raise ValueError(f"Unknown method: {method}")


def get_cache_subdir(method: str) -> str:
    """Get cache subdirectory for a method."""
    if method in {"opendde", "mol1"}:
        return method
    if method == "alphafold2":
        return "alphafold2"
    elif method == "esmfold2":
        return "esmfold2"
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# Lightweight Cache Checker
# =============================================================================

cache_checker_image = Image.debian_slim(python_version="3.12")


@app.function(image=cache_checker_image, volumes={"/cache": CACHE_VOLUME}, timeout=60)
def check_cache(method: str, params: dict[str, Any]) -> list | None:
    """Check if cached result exists and return it, or None if cache miss.

    This is a lightweight CPU-only function that avoids spinning up GPU containers
    just to check if a result is already cached.
    """
    from pathlib import Path

    cache_key = get_cache_key(method, params)
    cache_subdir = get_cache_subdir(method)
    cache_path = Path(f"/cache/{cache_subdir}/{cache_key}")
    cache_marker = cache_path / "_COMPLETE"

    if not cache_marker.exists():
        print(f"[cache-check] {method} MISS: {cache_key}")
        return None

    print(f"[cache-check] {method} HIT: {cache_key}")
    CACHE_VOLUME.reload()

    return [
        (str(f.relative_to(cache_path)), f.read_bytes())
        for f in cache_path.glob("**/*")
        if f.is_file() and f.name != "_COMPLETE"
    ]
