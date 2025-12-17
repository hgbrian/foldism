"""Chai-1 structure prediction with Modal Volume caching.

## Example FASTA (test.faa):
```
>protein|name=insulin
MAWTPLLLLLLSHCTGSLSQPVLTQPTSLSASPGASARFTCTLRSGINVGTY
>ligand|name=caffeine
CN1C=NC2=C1C(=O)N(C)C(=O)N2C
```

## Usage:
```
uv run modal run modal_chai1.py --input-faa test.faa
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

ENTITY_TYPES = {"protein", "ligand", "rna", "dna"}
CACHE_VERSION = "v1"

# Modal Volume for cache
CHAI1_CACHE_VOLUME = Volume.from_name("foldism-chai1-cache", create_if_missing=True)
CACHE_DIR = "/cache"


class Chai1Params(TypedDict, total=False):
    """Parameters for Chai-1 inference."""
    input_str: str  # Required: fasta content
    input_name: str  # Optional: filename (default: input.faa)
    num_trunk_recycles: int  # Optional: default 3
    num_diffn_timesteps: int  # Optional: default 200
    seed: int  # Optional: default 42
    use_esm_embeddings: bool  # Optional: default True
    use_msa_server: bool  # Optional: default True


def download_models():
    """Force download of Chai-1 models."""
    import torch
    from chai_lab.chai1 import run_inference

    Path(in_dir := "/tmp/in_dm").mkdir(parents=True, exist_ok=True)
    Path(out_dir := "/tmp/out_dm").mkdir(parents=True, exist_ok=True)

    with open(Path(in_dir) / "tmp.faa", "w") as out:
        out.write(">protein|name=pro\nMAWTPLLLLLLSHCTGSLSQPVLTQPTSL\n>ligand|name=lig\nCC\n")

    _ = run_inference(
        fasta_file=Path(in_dir) / "tmp.faa",
        output_dir=Path(out_dir),
        num_trunk_recycles=1,
        num_diffn_timesteps=10,
        seed=1,
        device=torch.device("cuda:0"),
        use_esm_embeddings=True,
    )


image = (
    Image.debian_slim()
    .apt_install("git", "wget")
    .pip_install("polars==1.19.0")
    .pip_install("git+https://github.com/chaidiscovery/chai-lab.git")
    .run_function(download_models, gpu="A10G")
)

app = App("chai1", image=image)


def _generate_cache_key(params: dict[str, Any]) -> str:
    """Generate deterministic cache key from params dict."""
    input_hash = sha256(params.get("input_str", "").encode()).hexdigest()
    cache_params = {
        "version": CACHE_VERSION,
        "input_hash": input_hash,
        "num_trunk_recycles": params.get("num_trunk_recycles", 3),
        "num_diffn_timesteps": params.get("num_diffn_timesteps", 200),
        "seed": params.get("seed", 42),
        "use_esm_embeddings": params.get("use_esm_embeddings", True),
        "use_msa_server": params.get("use_msa_server", True),
    }
    params_str = json.dumps(cache_params, sort_keys=True)
    return sha256(params_str.encode()).hexdigest()[:16]


@app.function(
    timeout=TIMEOUT * 60,
    gpu=GPU,
    volumes={CACHE_DIR: CHAI1_CACHE_VOLUME},
)
def chai1_from_file(
    params: dict[str, Any],
    overwrite: bool = False,
) -> list:
    """Run Chai-1 on a fasta file and return outputs.

    Args:
        params: Chai1Params dict with keys:
            - input_str (required): Fasta content in Chai-1 format
            - input_name: Filename (default: input.faa)
            - num_trunk_recycles: Number of trunk recycles (default: 3)
            - num_diffn_timesteps: Number of diffusion timesteps (default: 200)
            - seed: Random seed (default: 42)
            - use_esm_embeddings: Use ESM embeddings (default: True)
            - use_msa_server: Use MSA server (default: True)
        overwrite: If True, bypass cache and recompute

    Returns:
        List of (relative_path, bytes) tuples for output files.
    """
    from tempfile import TemporaryDirectory

    import torch
    from chai_lab.chai1 import run_inference

    input_str = params["input_str"]
    input_name = params.get("input_name", "input.faa")
    num_trunk_recycles = params.get("num_trunk_recycles", 3)
    num_diffn_timesteps = params.get("num_diffn_timesteps", 200)
    seed = params.get("seed", 42)
    use_esm_embeddings = params.get("use_esm_embeddings", True)
    use_msa_server = params.get("use_msa_server", True)

    # Generate cache key
    cache_key = _generate_cache_key(params)
    cache_path = Path(f"{CACHE_DIR}/{cache_key}")
    cache_marker = cache_path / "_COMPLETE"

    # Check cache
    if not overwrite and cache_marker.exists():
        print(f"Cache hit: {cache_key}")
        CHAI1_CACHE_VOLUME.reload()
        return [
            (out_file.relative_to(cache_path), out_file.read_bytes())
            for out_file in cache_path.glob("**/*")
            if out_file.is_file() and out_file.name != "_COMPLETE"
        ]

    print(f"Cache miss: {cache_key} - running inference")

    with TemporaryDirectory() as in_dir, TemporaryDirectory() as out_dir:
        fasta_path = Path(in_dir) / input_name
        fasta_path.write_text(input_str)

        _ = run_inference(
            fasta_file=Path(fasta_path),
            output_dir=Path(out_dir),
            num_trunk_recycles=num_trunk_recycles,
            num_diffn_timesteps=num_diffn_timesteps,
            seed=seed,
            device=torch.device("cuda:0"),
            use_esm_embeddings=use_esm_embeddings,
            use_msa_server=use_msa_server,
        )

        # Collect output files
        outputs = []
        for out_file in Path(out_dir).rglob("*"):
            if out_file.is_file():
                rel_path = out_file.relative_to(out_dir)
                outputs.append((rel_path, out_file.read_bytes()))

        # Save to cache
        cache_path.mkdir(parents=True, exist_ok=True)
        for rel_path, content in outputs:
            out_path = cache_path / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(content)

        cache_marker.touch()
        CHAI1_CACHE_VOLUME.commit()

        return outputs


@app.local_entrypoint()
def main(
    input_faa: str,
    out_dir: str = "./out/chai1",
    overwrite: bool = False,
):
    """Run Chai-1 structure prediction.

    Args:
        input_faa: Path to input FASTA file
        out_dir: Output directory
        overwrite: Bypass cache if True
    """
    input_content = open(input_faa).read()
    run_name = Path(input_faa).stem

    outputs = chai1_from_file.remote(
        params={"input_str": input_content, "input_name": f"{run_name}.faa"},
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
