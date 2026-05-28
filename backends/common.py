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
TIMEOUT = int(os.environ.get("TIMEOUT", 60))  # 60 minutes default
CACHE_VERSION = "v2"  # v2: per-chain paired MSA, AF2 single_sequence mode, boltz MSA source in key, protenix server-fallback removed, unified.json schema

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
    "alphafold2": FoldingApp("AlphaFold2", "alphafold2", "AlphaFold2/ColabFold", "af2_fasta"),
    "openfold3": FoldingApp("OpenFold 3", "openfold3", "OpenFold 3 (AlphaFold3-style)", "openfold3_fasta"),
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


def _fasta_to_boltz_yaml(
    fasta_str: str,
    msa_paths_per_chain: list[str | None] | None = None,
) -> str:
    """Convert FASTA → Boltz YAML.

    Recognizes both header styles:
      - "X|protein", "X|dna", "X|rna"  (chain_id|type, all backends)
      - "protein|...", "ligand|...", "dna|...", "rna|..."  (type-first; same
        convention used by chai/protenix/openfold3/af2)
    Falls back to "protein" + sequential chain id (A, B, …) when neither match.

    `msa_paths_per_chain` is indexed by PROTEIN-CHAIN POSITION so homomers
    can carry distinct paired-MSA CSVs per chain.
    """
    import yaml

    chains = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    yaml_dict: dict[str, Any] = {"sequences": []}
    chain_first = re.compile(r"^([A-Z])\|(protein|dna|rna)$", re.IGNORECASE)
    type_first = {"protein", "dna", "rna", "ligand"}
    protein_chain_idx = 0

    for n, (seq_id, seq) in enumerate(fasta_iter(fasta_str)):
        s_info = chain_first.match(seq_id.strip())
        first_part = seq_id.split("|")[0].lower() if "|" in seq_id else ""
        if s_info:
            entity_type = s_info.group(2).lower()
            chain_id = s_info.group(1).upper()
        elif first_part in type_first:
            entity_type = first_part
            chain_id = chains[n] if n < len(chains) else f"chain_{n}"
        else:
            entity_type = "protein"
            chain_id = chains[n] if n < len(chains) else f"chain_{n}"
        if entity_type == "protein":
            assert all(aa.upper() in ALLOWED_AAS for aa in seq), f"Invalid AAs: {seq}"
        chain_dict: dict[str, Any] = {"id": chain_id}
        if entity_type == "ligand":
            chain_dict["smiles"] = seq
        else:
            chain_dict["sequence"] = seq
        if entity_type == "protein":
            if msa_paths_per_chain and protein_chain_idx < len(msa_paths_per_chain):
                p = msa_paths_per_chain[protein_chain_idx]
                if p:
                    chain_dict["msa"] = p
            protein_chain_idx += 1
        entity = {entity_type: chain_dict}
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


def _fasta_to_protenix_json(
    input_faa: str,
    name: str = "input",
    msa_result: dict | None = None,
) -> str:
    """Convert FASTA → Protenix input JSON.

    `msa_result["paired_per_chain"]`, when present, is a list indexed by
    PROTEIN-CHAIN POSITION (so homomers carry distinct paired MSAs).
    `msa_result["unpaired"]` remains a dict keyed by sequence (one MSA per
    unique seq, since identical seqs share unpaired databases).
    """
    sequences = []
    paired_per_chain_list = (msa_result or {}).get("paired_per_chain") if msa_result else None
    protein_chain_idx = 0

    for seq_id, seq in fasta_iter(input_faa):
        parts = seq_id.split("|")
        first_part = parts[0].lower() if parts else ""

        if first_part in PROTENIX_ENTITY_MAP:
            entity_type = first_part
        else:
            entity_type = "protein"

        protenix_type = PROTENIX_ENTITY_MAP[entity_type]
        if entity_type == "ligand":
            entity = {protenix_type: {"ligand": seq, "count": 1}}
        else:
            chain_dict: dict[str, Any] = {"sequence": seq, "count": 1}
            if entity_type == "protein":
                if msa_result and seq in msa_result.get("unpaired", {}):
                    chain_dict["unpairedMsaPath"] = f"{msa_result['unpaired'][seq]}/merged.a3m"
                    if paired_per_chain_list and protein_chain_idx < len(paired_per_chain_list):
                        pp = paired_per_chain_list[protein_chain_idx]
                        if pp:
                            chain_dict["pairedMsaPath"] = pp
                    elif msa_result.get("paired_dir"):
                        chain_dict["pairedMsaPath"] = f"{msa_result['paired_dir']}/pair.a3m"
                protein_chain_idx += 1
            entity = {protenix_type: chain_dict}
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


def _fasta_to_openfold3_json(
    fasta_str: str,
    name: str = "input",
    msa_dir_map: dict[str, str] | None = None,
    paired_dir_per_chain: list[str | None] | None = None,
) -> str:
    """Convert FASTA to OpenFold 3 query JSON format.

    `msa_dir_map` (seq → dir) supplies the unpaired/main MSA (same MSA for
    identical sequences). `paired_dir_per_chain` is indexed by PROTEIN-CHAIN
    POSITION (0, 1, …) so homomers can carry distinct paired MSAs per chain.
    """
    chain_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    chains_list: list[dict[str, Any]] = []
    protein_chain_idx = 0

    for n, (seq_id, seq) in enumerate(fasta_iter(fasta_str)):
        parts = seq_id.split("|")
        first_part = parts[0].lower() if parts else ""

        if first_part in ("protein", "dna", "rna", "ligand"):
            entity_type = first_part
        else:
            entity_type = "protein"

        cid = chain_letters[n] if n < len(chain_letters) else f"chain_{n}"

        chain: dict[str, Any] = {
            "molecule_type": entity_type,
            "chain_ids": [cid],
        }

        if entity_type == "ligand":
            chain["smiles"] = seq
        else:
            chain["sequence"] = seq

        if entity_type == "protein":
            if msa_dir_map and seq in msa_dir_map:
                chain["main_msa_file_paths"] = msa_dir_map[seq] + "/colabfold_main.a3m"
            if paired_dir_per_chain and protein_chain_idx < len(paired_dir_per_chain):
                pdir = paired_dir_per_chain[protein_chain_idx]
                if pdir:
                    chain["paired_msa_file_paths"] = pdir + "/colabfold_paired.a3m"
            protein_chain_idx += 1

        chains_list.append(chain)

    query: dict[str, Any] = {"chains": chains_list}
    if msa_dir_map:
        query["use_msas"] = True

    return json.dumps({"queries": {name: query}}, indent=2)


CONVERTERS = {
    "boltz_yaml": _fasta_to_boltz_yaml,
    "chai_fasta": _fasta_to_chai_fasta,
    "protenix_fasta": _normalize_fasta,
    "af2_fasta": _fasta_to_af2_fasta,
    "openfold3_fasta": _normalize_fasta,
    "esmfold2_fasta": _normalize_fasta,
}


def convert_for_app(fasta_str: str, app_key: str) -> str:
    app_def = FOLDING_APPS[app_key]
    return CONVERTERS[app_def.input_format](fasta_str)


# =============================================================================
# Cache Key Functions
# =============================================================================

BOLTZ_BASE_PARAMS = "--seed 42 --no_kernels --recycling_steps 3 --step_scale 1.0 --diffusion_samples 5"


def boltz_cache_key(params: dict[str, Any]) -> str:
    """Generate cache key for Boltz-2 predictions.

    `use_msa` distinguishes MSA SOURCE, not whether MSA is used (boltz always
    uses MSA): True means we supply pre-fetched ColabSearch MSAs; False means
    boltz falls back to its built-in MMseqs2 server. The two paths produce
    different MSAs and therefore different structures, so they must cache
    separately.
    """
    from .boltz import BOLTZ_PKG
    input_hash = sha256(params.get("input_str", "").encode()).hexdigest()
    cache_params = {
        "version": CACHE_VERSION,
        "input_hash": input_hash,
        "params_str": params.get("params_str", BOLTZ_BASE_PARAMS),
        "use_msa": params.get("use_msa", True),
        "pkg": BOLTZ_PKG,
    }
    return sha256(json.dumps(cache_params, sort_keys=True).encode()).hexdigest()[:16]


def chai1_cache_key(params: dict[str, Any]) -> str:
    """Generate cache key for Chai-1 predictions."""
    from .chai1 import CHAI1_GIT_REF
    input_hash = sha256(params.get("input_str", "").encode()).hexdigest()
    cache_params = {
        "version": CACHE_VERSION,
        "input_hash": input_hash,
        "num_trunk_recycles": params.get("num_trunk_recycles", 3),
        "num_diffn_timesteps": params.get("num_diffn_timesteps", 200),
        "seed": params.get("seed", 42),
        "use_msa_server": params.get("use_msa_server", True),
        "chai_lab_rev": CHAI1_GIT_REF,
    }
    return sha256(json.dumps(cache_params, sort_keys=True).encode()).hexdigest()[:16]


def protenix_cache_key(params: dict[str, Any]) -> str:
    """Generate cache key for Protenix predictions."""
    from .protenix import PROTENIX_BASE_MODEL, PROTENIX_HF_BUCKET, PROTENIX_HF_REVISION
    input_hash = sha256(params.get("input_str", "").encode()).hexdigest()
    cache_params = {
        "version": CACHE_VERSION,
        "input_hash": input_hash,
        "seeds": params.get("seeds", "42"),
        "use_msa": params.get("use_msa", True),
        "model": PROTENIX_BASE_MODEL,
        "weights_src": f"{PROTENIX_HF_BUCKET}@{PROTENIX_HF_REVISION}",
        # Bumped after the --use_msa CLI-flag fix (was hardcoded "false", now
        # passes "true" so unpairedMsaPath in the JSON is actually honored).
        # Old entries cached MSA-less predictions under use_msa=True keys.
        "cli_wiring": "v2",
    }
    return sha256(json.dumps(cache_params, sort_keys=True).encode()).hexdigest()[:16]


def alphafold_cache_key(params: dict[str, Any]) -> str:
    """Generate cache key for AlphaFold2 predictions."""
    from .alphafold2 import COLABFOLD_GIT_REF
    input_hash = sha256(params.get("input_str", "").encode()).hexdigest()
    # `use_msa=True` means a pre-fetched colabsearch MSA is supplied;
    # `use_msa=False` lets colabfold use single_sequence mode.
    # These produce different outputs and must cache separately.
    cache_params = {
        "version": CACHE_VERSION,
        "input_hash": input_hash,
        "models": params.get("models", [1]),
        "num_recycles": params.get("num_recycles", 3),
        "use_msa": params.get("use_msa", True),
        "colabfold_rev": COLABFOLD_GIT_REF,
    }
    return sha256(json.dumps(cache_params, sort_keys=True).encode()).hexdigest()[:16]


def openfold3_cache_key(params: dict[str, Any]) -> str:
    """Generate cache key for OpenFold 3 predictions."""
    from .openfold3 import OPENFOLD3_CHECKPOINT
    input_hash = sha256(params.get("input_str", "").encode()).hexdigest()
    cache_params = {
        "version": CACHE_VERSION,
        "input_hash": input_hash,
        "seed": params.get("seed", 42),
        "use_msa": params.get("use_msa", True),
        "checkpoint": OPENFOLD3_CHECKPOINT,
    }
    return sha256(json.dumps(cache_params, sort_keys=True).encode()).hexdigest()[:16]


def esmfold2_cache_key(params: dict[str, Any]) -> str:
    """Generate cache key for ESMFold2 predictions."""
    from .esmfold2 import ESMFOLD2_HF_REPO, ESMFOLD2_HF_REVISION
    input_hash = sha256(params.get("input_str", "").encode()).hexdigest()
    cache_params = {
        "version": CACHE_VERSION,
        "input_hash": input_hash,
        "seed": params.get("seed", 42),
        "num_diffusion_samples": params.get("num_diffusion_samples", 1),
        "num_sampling_steps": params.get("num_sampling_steps", 50),
        "num_loops": params.get("num_loops", 3),
        "use_msa": params.get("use_msa", True),
        "model": "esmfold2",
        "checkpoint": f"{ESMFOLD2_HF_REPO}@{ESMFOLD2_HF_REVISION}",
        # Bumped when scores.json schema grows (added ptm/iptm/chain_pair_iptm).
        # Old cached entries had only plddt fields and would mask the new metrics.
        "scores_v": "v2",
    }
    return sha256(json.dumps(cache_params, sort_keys=True).encode()).hexdigest()[:16]


def get_cache_key(method: str, params: dict[str, Any]) -> str:
    """Get cache key for any method."""
    if method == "boltz2":
        return boltz_cache_key(params)
    elif method == "chai1":
        return chai1_cache_key(params)
    elif method == "protenix":
        return protenix_cache_key(params)
    elif method == "alphafold2":
        return alphafold_cache_key(params)
    elif method == "openfold3":
        return openfold3_cache_key(params)
    elif method == "esmfold2":
        return esmfold2_cache_key(params)
    else:
        raise ValueError(f"Unknown method: {method}")


def get_cache_subdir(method: str) -> str:
    """Get cache subdirectory for a method."""
    if method == "boltz2":
        return "boltz2"
    elif method == "chai1":
        return "chai1"
    elif method == "protenix":
        return "protenix"
    elif method == "alphafold2":
        return "alphafold2"
    elif method == "openfold3":
        return "openfold3"
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
