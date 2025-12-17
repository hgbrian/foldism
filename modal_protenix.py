"""Protenix (AlphaFold3-style) structure prediction with Modal Volume caching.

Protenix is an open-source PyTorch reproduction of AlphaFold 3.

## Example FASTA (test.faa):
```
>protein|A
MAWTPLLLLLLSHCTGSLSQPVLTQPTSLSASPGASARFTCTLRSGINVGTY
```

## Usage:
```
uv run modal run modal_protenix.py --input-faa test.faa
```

## With MSA:
```
uv run modal run modal_protenix.py --input-faa test.faa --use-msa
```
"""

import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Any, TypedDict

from modal import App, Image, Volume

GPU = os.environ.get("GPU", "L40S")
TIMEOUT = int(os.environ.get("TIMEOUT", 60))

ENTITY_TYPES = {"protein", "dna", "rna", "ligand", "ion"}
CACHE_VERSION = "v1"

# Modal Volume for cache
PROTENIX_CACHE_VOLUME = Volume.from_name("foldism-protenix-cache", create_if_missing=True)
CACHE_DIR = "/cache"

# Entity type mapping: user-friendly names -> Protenix JSON format
ENTITY_TYPE_MAP = {
    "protein": "proteinChain",
    "dna": "dnaSequence",
    "rna": "rnaSequence",
    "ligand": "ligand",
    "ion": "ion",
}

DEFAULT_MODEL = "protenix_base_default_v0.5.0"
DEFAULT_SEEDS = "42"


class ProtenixParams(TypedDict, total=False):
    """Parameters for Protenix inference."""
    input_str: str  # Required: FASTA or JSON content
    input_name: str  # Optional: name for prediction job (default: input)
    seeds: str  # Optional: comma-separated seeds (default: 42)
    use_msa: bool  # Optional: use MSA (default: True)
    model_name: str  # Optional: model name (default: protenix_base_default_v0.5.0)


def fasta_iter(fasta_str: str):
    """Yield (seq_id, seq) tuples from a FASTA string."""
    from io import StringIO
    from itertools import groupby

    with StringIO(fasta_str) as fh:
        faiter = (x[1] for x in groupby(fh, lambda line: line.startswith(">")))
        for header in faiter:
            header = next(header)[1:].strip()
            seq = "".join(s.strip() for s in next(faiter))
            yield header, seq


def download_models():
    """Force download of Protenix models."""
    from subprocess import run
    from tempfile import TemporaryDirectory

    Path(out_dir := "/tmp/out_dm").mkdir(parents=True, exist_ok=True)

    test_json = """[
    {
        "name": "test",
        "sequences": [
            {
                "proteinChain": {
                    "sequence": "MAWTPLLLLLLSHCTGSLSQPVLTQPTSL",
                    "count": 1
                }
            }
        ]
    }
]"""

    with TemporaryDirectory() as td:
        json_path = Path(td) / "test.json"
        json_path.write_text(test_json)

        print("Downloading base model...")
        run(
            f'protenix predict --input "{json_path}" --out_dir "{out_dir}" --seeds 42 --use_msa false',
            shell=True,
            check=True,
        )

    print("Model download complete")


image = (
    Image.debian_slim()
    .apt_install("git", "wget")
    .pip_install("polars==1.19.0")
    .pip_install("protenix")
    .run_function(download_models, gpu="A10G")
)

app = App("protenix", image=image)


def _fasta_to_json(input_faa: str, name: str = "input") -> str:
    """Convert FASTA to Protenix JSON format."""
    import re

    rx = re.compile(r"^([^|]+)\|([^|]*)\|?(.*)$")
    sequences = []

    for seq_id, seq in fasta_iter(input_faa):
        match = rx.search(seq_id)
        if match:
            entity_type = match.group(1).lower()
        else:
            entity_type = "protein"

        if entity_type not in ENTITY_TYPES:
            raise ValueError(f"Invalid entity type: {entity_type}. Must be one of {ENTITY_TYPES}")

        protenix_entity_type = ENTITY_TYPE_MAP[entity_type]

        if entity_type == "ligand":
            entity = {protenix_entity_type: {"ligand": seq, "count": 1}}
        else:
            entity = {protenix_entity_type: {"sequence": seq, "count": 1}}

        sequences.append(entity)

    prediction = [{"name": name, "sequences": sequences}]
    return json.dumps(prediction, indent=2)


def _generate_cache_key(params: dict[str, Any]) -> str:
    """Generate deterministic cache key from params dict."""
    input_hash = sha256(params.get("input_str", "").encode()).hexdigest()
    cache_params = {
        "version": CACHE_VERSION,
        "input_hash": input_hash,
        "seeds": params.get("seeds", DEFAULT_SEEDS),
        "use_msa": params.get("use_msa", True),
        "model_name": params.get("model_name", DEFAULT_MODEL),
    }
    params_str = json.dumps(cache_params, sort_keys=True)
    return sha256(params_str.encode()).hexdigest()[:16]


@app.function(
    timeout=TIMEOUT * 60,
    gpu=GPU,
    volumes={CACHE_DIR: PROTENIX_CACHE_VOLUME},
)
def protenix_from_file(
    params: dict[str, Any],
    overwrite: bool = False,
) -> list:
    """Run Protenix on input and return output files.

    Args:
        params: ProtenixParams dict with keys:
            - input_str (required): FASTA or JSON content
            - input_name: name for prediction job (default: input)
            - seeds: comma-separated seeds (default: 42)
            - use_msa: use MSA (default: True)
            - model_name: model name (default: protenix_base_default_v0.5.0)
        overwrite: If True, bypass cache and recompute

    Returns:
        List of (relative_path, bytes) tuples for output files.
    """
    import subprocess
    from tempfile import TemporaryDirectory

    input_str = params["input_str"]
    input_name = params.get("input_name", "input")
    seeds = params.get("seeds", DEFAULT_SEEDS)
    use_msa = params.get("use_msa", True)

    # Generate cache key
    cache_key = _generate_cache_key(params)
    cache_path = Path(f"{CACHE_DIR}/{cache_key}")
    cache_marker = cache_path / "_COMPLETE"

    # Check cache
    if not overwrite and cache_marker.exists():
        print(f"Cache hit: {cache_key}")
        PROTENIX_CACHE_VOLUME.reload()
        return [
            (out_file.relative_to(cache_path), out_file.read_bytes())
            for out_file in cache_path.glob("**/*")
            if out_file.is_file() and out_file.name != "_COMPLETE"
        ]

    print(f"Cache miss: {cache_key} - running inference")

    # Convert FASTA to JSON if needed
    if input_str.strip().startswith(">"):
        input_str = _fasta_to_json(input_str, input_name)

    with TemporaryDirectory() as in_dir, TemporaryDirectory() as out_dir:
        json_path = Path(in_dir) / "input.json"
        json_path.write_text(input_str)

        use_msa_str = "true" if use_msa else "false"
        cmd = f'protenix predict --input "{json_path}" --out_dir "{out_dir}" --seeds {seeds} --use_msa {use_msa_str}'
        print(f"Running: {cmd}")

        subprocess.run(cmd, shell=True, check=True)

        # Find prediction directory
        pred_dir = Path(out_dir) / input_name
        if not pred_dir.exists():
            pred_dir = Path(out_dir)

        # Collect output files
        outputs = []
        for out_file in pred_dir.rglob("*"):
            if out_file.is_file():
                rel_path = out_file.relative_to(pred_dir)
                outputs.append((rel_path, out_file.read_bytes()))

        # Save to cache
        cache_path.mkdir(parents=True, exist_ok=True)
        for rel_path, content in outputs:
            out_path = cache_path / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(content)

        cache_marker.touch()
        PROTENIX_CACHE_VOLUME.commit()

        return outputs


@app.local_entrypoint()
def main(
    input_faa: str,
    seeds: str = DEFAULT_SEEDS,
    use_msa: bool = True,
    out_dir: str = "./out/protenix",
    overwrite: bool = False,
):
    """Run Protenix structure prediction.

    Args:
        input_faa: Path to input FASTA file
        seeds: Comma-separated random seeds
        use_msa: Use MSA server
        out_dir: Output directory
        overwrite: Bypass cache if True
    """
    input_content = open(input_faa).read()
    run_name = Path(input_faa).stem

    outputs = protenix_from_file.remote(
        params={
            "input_str": input_content,
            "input_name": run_name,
            "seeds": seeds,
            "use_msa": use_msa,
        },
        overwrite=overwrite,
    )

    out_path = Path(out_dir) / run_name
    out_path.mkdir(parents=True, exist_ok=True)

    for rel_path, content in outputs:
        file_path = out_path / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)
        print(f"  -> {file_path}")

    print(f"\nOutput directory: {out_path}")
