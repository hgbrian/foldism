"""Boltz-2 structure prediction with Modal Volume caching.

## Example input (test.yaml):
```yaml
sequences:
  - protein:
      id: A
      sequence: TDKLIFGKGTRVTVEP
```

## Usage:
```
uv run modal run modal_boltz.py --input-yaml test.yaml
```

## High-quality params (default):
```
uv run modal run modal_boltz.py --input-yaml test.yaml --params-str "--use_msa_server --seed 42 --no_kernels --recycling_steps 10 --step_scale 1.0 --diffusion_samples 10"
```
"""

import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Any, TypedDict

from modal import App, Image, Volume

GPU = os.environ.get("GPU", "L40S")
TIMEOUT = int(os.environ.get("TIMEOUT", 30))

ENTITY_TYPES = {"protein", "dna", "rna", "ccd", "smiles"}
ALLOWED_AAS = "ACDEFGHIKLMNPQRSTVWY"
CACHE_VERSION = "v1"

# Modal Volumes for models and cache
BOLTZ_MODEL_VOLUME = Volume.from_name("foldism-boltz-models", create_if_missing=True)
BOLTZ_CACHE_VOLUME = Volume.from_name("foldism-boltz-cache", create_if_missing=True)

MODEL_CACHE_DIR = "/models"
CACHE_DIR = "/cache"

DEFAULT_PARAMS = "--use_msa_server --seed 42 --no_kernels --recycling_steps 10 --step_scale 1.0 --diffusion_samples 10"


class BoltzParams(TypedDict, total=False):
    """Parameters for Boltz inference."""
    input_str: str  # Required: yaml or fasta content
    params_str: str  # Optional: CLI params (default: DEFAULT_PARAMS)


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


def download_model():
    """Force download of the Boltz model."""
    from boltz.main import download_boltz1, download_boltz2

    if not Path(f"{MODEL_CACHE_DIR}/boltz1_conf.ckpt").exists():
        print("downloading boltz 1")
        download_boltz1(Path(MODEL_CACHE_DIR))

    if not Path(f"{MODEL_CACHE_DIR}/boltz2_conf.ckpt").exists():
        print("downloading boltz 2")
        download_boltz2(Path(MODEL_CACHE_DIR))


image = (
    Image.micromamba()
    .apt_install("wget", "git", "gcc", "g++")
    .pip_install(
        "colabfold[alphafold-minus-jax]@git+https://github.com/sokrypton/ColabFold@acc0bf772f22feb7f887ad132b7313ff415c8a9f"
    )
    .micromamba_install(
        "kalign2=2.04", "hhsuite=3.3.0", channels=["conda-forge", "bioconda"]
    )
    .run_commands(
        'pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html',
        gpu="a10g",
    )
    .run_commands("python -m colabfold.download")
    .apt_install("build-essential")
    .pip_install("polars==1.19.0", "boltz==2.2.0")
    .run_function(
        download_model,
        gpu="a10g",
        volumes={MODEL_CACHE_DIR: BOLTZ_MODEL_VOLUME},
    )
)

app = App("boltz", image=image)


def _fasta_to_yaml(input_faa: str) -> str:
    """Convert FASTA to Boltz YAML format."""
    import re
    import yaml

    chains = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    yaml_dict = {"sequences": []}

    rx = re.compile(r"^([A-Z])\|(\S+)\|(.*)$")

    for n, (seq_id, seq) in enumerate(fasta_iter(input_faa)):
        if n >= len(chains):
            raise NotImplementedError(">26 chains not supported")

        s_info = rx.search(seq_id)
        if s_info is not None:
            entity_type = s_info.groups()[1].lower()
            if entity_type not in ["protein", "dna", "rna"]:
                raise NotImplementedError(f"Entity type {entity_type} not supported. Use YAML.")
            chain_id = s_info.groups()[0].upper()
        else:
            entity_type = "protein"
            chain_id = chains[n]

        if entity_type == "protein":
            assert all(aa.upper() in ALLOWED_AAS for aa in seq), f"Invalid AAs: {seq}"

        entity = {entity_type: {"id": chain_id, "sequence": seq}}
        yaml_dict["sequences"].append(entity)

    print(yaml.dump(yaml_dict, sort_keys=False))
    return yaml.dump(yaml_dict, sort_keys=False)


def _generate_cache_key(params: dict[str, Any]) -> str:
    """Generate deterministic cache key from params dict."""
    input_hash = sha256(params.get("input_str", "").encode()).hexdigest()
    cache_params = {
        "version": CACHE_VERSION,
        "input_hash": input_hash,
        "params_str": params.get("params_str", DEFAULT_PARAMS),
    }
    params_str = json.dumps(cache_params, sort_keys=True)
    return sha256(params_str.encode()).hexdigest()[:16]


@app.function(
    timeout=TIMEOUT * 60,
    gpu=GPU,
    volumes={
        MODEL_CACHE_DIR: BOLTZ_MODEL_VOLUME,
        CACHE_DIR: BOLTZ_CACHE_VOLUME,
    },
)
def boltz_from_file(
    params: dict[str, Any],
    overwrite: bool = False,
) -> list:
    """Run Boltz on input and return output files.

    Args:
        params: BoltzParams dict with keys:
            - input_str (required): YAML or FASTA content
            - params_str: CLI params (default: DEFAULT_PARAMS)
        overwrite: If True, bypass cache and recompute

    Returns:
        List of (relative_path, bytes) tuples for output files.
    """
    import subprocess
    from tempfile import TemporaryDirectory

    input_str = params["input_str"]
    params_str = params.get("params_str", DEFAULT_PARAMS)

    # Generate cache key
    cache_key = _generate_cache_key(params)
    cache_path = Path(f"{CACHE_DIR}/{cache_key}")
    cache_marker = cache_path / "_COMPLETE"

    # Check cache
    if not overwrite and cache_marker.exists():
        print(f"Cache hit: {cache_key}")
        BOLTZ_CACHE_VOLUME.reload()
        return [
            (out_file.relative_to(cache_path), out_file.read_bytes())
            for out_file in cache_path.glob("**/*")
            if out_file.is_file() and out_file.name != "_COMPLETE"
        ]

    print(f"Cache miss: {cache_key} - running inference")

    # Convert FASTA to YAML if needed
    if input_str.strip().startswith(">"):
        input_str = _fasta_to_yaml(input_str)

    with TemporaryDirectory() as in_dir, TemporaryDirectory() as out_dir:
        input_path = Path(in_dir) / "input.yaml"
        input_path.write_text(input_str)

        cmd = f'boltz predict {input_path} --out_dir {out_dir} --cache {MODEL_CACHE_DIR} {params_str}'
        print(f"Running: {cmd}")

        subprocess.run(cmd, shell=True, check=True)

        # Find prediction directory
        pred_dir = None
        for d in Path(out_dir).iterdir():
            if d.is_dir() and d.name.startswith("boltz_results"):
                pred_dir = d
                break

        if pred_dir is None:
            raise RuntimeError(f"No prediction directory found in {out_dir}")

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
        BOLTZ_CACHE_VOLUME.commit()

        return outputs


@app.local_entrypoint()
def main(
    input_yaml: str,
    params_str: str = DEFAULT_PARAMS,
    out_dir: str = "./out/boltz",
    overwrite: bool = False,
):
    """Run Boltz structure prediction.

    Args:
        input_yaml: Path to input YAML file
        params_str: Boltz CLI parameters
        out_dir: Output directory
        overwrite: Bypass cache if True
    """
    input_content = open(input_yaml).read()
    run_name = Path(input_yaml).stem

    outputs = boltz_from_file.remote(
        params={"input_str": input_content, "params_str": params_str},
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
