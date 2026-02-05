"""Common utilities and shared resources for all backends."""

from __future__ import annotations

import json
import os
import re
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
TIMEOUT = int(os.environ.get("TIMEOUT", 15))  # 15 minutes default
CACHE_VERSION = "v1"

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
    "boltz2": FoldingApp("Boltz-2", "boltz2", "Boltz-2 structure prediction", "boltz_yaml"),
    "chai1": FoldingApp("Chai-1", "chai1", "Chai-1 structure prediction", "chai_fasta"),
    "protenix": FoldingApp("Protenix", "protenix", "Protenix (AlphaFold3-style)", "protenix_fasta"),
    "protenix-mini": FoldingApp("Protenix-Mini", "protenix-mini", "Protenix Mini (fast)", "protenix_fasta"),
    "alphafold2": FoldingApp("AlphaFold2", "alphafold2", "AlphaFold2/ColabFold", "af2_fasta"),
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


# =============================================================================
# Format Converters
# =============================================================================


def _fasta_to_boltz_yaml(fasta_str: str) -> str:
    import yaml

    chains = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    yaml_dict: dict[str, Any] = {"sequences": []}
    # Match patterns like "A|protein" at the START of header only
    rx = re.compile(r"^([A-Z])\|(protein|dna|rna)$", re.IGNORECASE)

    for n, (seq_id, seq) in enumerate(fasta_iter(fasta_str)):
        # Only match if the entire header is in the format "X|type"
        s_info = rx.match(seq_id.strip())
        if s_info:
            entity_type = s_info.group(2).lower()
            chain_id = s_info.group(1).upper()
        else:
            # Default to protein for all other header formats
            entity_type = "protein"
            chain_id = chains[n] if n < len(chains) else f"chain_{n}"
        if entity_type == "protein":
            assert all(aa.upper() in ALLOWED_AAS for aa in seq), f"Invalid AAs: {seq}"
        entity = {entity_type: {"id": chain_id, "sequence": seq}}
        yaml_dict["sequences"].append(entity)

    return yaml.dump(yaml_dict, sort_keys=False)


def _fasta_to_chai_fasta(fasta_str: str) -> str:
    """Convert FASTA to Chai format with deterministic chain names based on sequence hash."""
    valid_chai_types = {"protein", "ligand", "dna", "rna"}
    entries = []
    for seq_id, seq in fasta_iter(fasta_str):
        parts = seq_id.split("|")
        first_part = parts[0].lower() if parts else ""
        entity_type = first_part if first_part in valid_chai_types else "protein"
        entries.append((entity_type, seq))

    seq_hash = sha256(":".join(seq for _, seq in entries).encode()).hexdigest()[:6]
    lines = []
    for i, (entity_type, seq) in enumerate(entries):
        lines.append(f">{entity_type}|name={seq_hash}_{i}")
        lines.append(seq)
    return "\n".join(lines)


PROTENIX_ENTITY_MAP = {"protein": "proteinChain", "dna": "dnaSequence", "rna": "rnaSequence", "ligand": "ligand", "ion": "ion"}


def _fasta_to_protenix_json(input_faa: str, name: str = "input") -> str:
    sequences = []

    for seq_id, seq in fasta_iter(input_faa):
        # Check if header starts with a valid entity type
        parts = seq_id.split("|")
        first_part = parts[0].lower() if parts else ""

        if first_part in PROTENIX_ENTITY_MAP:
            entity_type = first_part
        else:
            # Default to protein for any unrecognized format
            entity_type = "protein"

        protenix_type = PROTENIX_ENTITY_MAP[entity_type]
        if entity_type == "ligand":
            entity = {protenix_type: {"ligand": seq, "count": 1}}
        else:
            entity = {protenix_type: {"sequence": seq, "count": 1}}
        sequences.append(entity)

    return json.dumps([{"name": name, "sequences": sequences}], indent=2)


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
    "boltz_yaml": _fasta_to_boltz_yaml,
    "chai_fasta": _fasta_to_chai_fasta,
    "protenix_fasta": _normalize_fasta,
    "af2_fasta": _fasta_to_af2_fasta,
}


def convert_for_app(fasta_str: str, app_key: str) -> str:
    app_def = FOLDING_APPS[app_key]
    return CONVERTERS[app_def.input_format](fasta_str)


# =============================================================================
# Cache Key Functions
# =============================================================================

BOLTZ_BASE_PARAMS = "--seed 42 --no_kernels --recycling_steps 3 --step_scale 1.0 --diffusion_samples 5"


def boltz_cache_key(params: dict[str, Any]) -> str:
    """Generate cache key for Boltz-2 predictions."""
    input_hash = sha256(params.get("input_str", "").encode()).hexdigest()
    # Boltz always uses MSA server - force True for cache consistency
    cache_params = {
        "version": CACHE_VERSION,
        "input_hash": input_hash,
        "params_str": params.get("params_str", BOLTZ_BASE_PARAMS),
        "use_msa": True,
    }
    return sha256(json.dumps(cache_params, sort_keys=True).encode()).hexdigest()[:16]


def chai1_cache_key(params: dict[str, Any]) -> str:
    """Generate cache key for Chai-1 predictions."""
    input_hash = sha256(params.get("input_str", "").encode()).hexdigest()
    cache_params = {
        "version": CACHE_VERSION,
        "input_hash": input_hash,
        "num_trunk_recycles": params.get("num_trunk_recycles", 3),
        "num_diffn_timesteps": params.get("num_diffn_timesteps", 200),
        "seed": params.get("seed", 42),
        "use_msa_server": params.get("use_msa_server", True),
    }
    return sha256(json.dumps(cache_params, sort_keys=True).encode()).hexdigest()[:16]


def protenix_cache_key(params: dict[str, Any]) -> str:
    """Generate cache key for Protenix predictions."""
    input_hash = sha256(params.get("input_str", "").encode()).hexdigest()
    model = params.get("model", "protenix")
    model_version = "v0.5.0" if model == "protenix_mini" else "v1.0.0"
    cache_params = {
        "version": CACHE_VERSION,
        "input_hash": input_hash,
        "seeds": params.get("seeds", "42"),
        "use_msa": params.get("use_msa", True),
        "model": model,
        "model_version": model_version,
    }
    return sha256(json.dumps(cache_params, sort_keys=True).encode()).hexdigest()[:16]


def alphafold_cache_key(params: dict[str, Any]) -> str:
    """Generate cache key for AlphaFold2 predictions."""
    input_hash = sha256(params.get("input_str", "").encode()).hexdigest()
    cache_params = {
        "version": CACHE_VERSION,
        "input_hash": input_hash,
        "models": params.get("models", [1]),
        "num_recycles": params.get("num_recycles", 3),
    }
    return sha256(json.dumps(cache_params, sort_keys=True).encode()).hexdigest()[:16]


def get_cache_key(method: str, params: dict[str, Any]) -> str:
    """Get cache key for any method."""
    if method == "boltz2":
        return boltz_cache_key(params)
    elif method == "chai1":
        return chai1_cache_key(params)
    elif method in ("protenix", "protenix-mini"):
        return protenix_cache_key(params)
    elif method == "alphafold2":
        return alphafold_cache_key(params)
    else:
        raise ValueError(f"Unknown method: {method}")


def get_cache_subdir(method: str) -> str:
    """Get cache subdirectory for a method."""
    if method == "boltz2":
        return "boltz2"
    elif method == "chai1":
        return "chai1"
    elif method in ("protenix", "protenix-mini"):
        return "protenix"
    elif method == "alphafold2":
        return "alphafold2"
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
