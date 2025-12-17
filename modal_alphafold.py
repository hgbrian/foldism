"""AlphaFold2/ColabFold structure prediction with Modal Volume caching.

## Example FASTA (test.fasta):
```
>protein
MAWTPLLLLLLSHCTGSLSQPVLTQPTSLSASPGASARFTCTLRSGINVGTY
```

## For complexes (chains joined with colons):
```
>complex
MAWTPLLLLLLSHCTGSLSQ:GVYDGREHTV
```

## Usage:
```
uv run modal run modal_alphafold.py --input-fasta test.fasta
```
"""

import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Any, TypedDict

from modal import App, Image, Volume

GPU = os.environ.get("GPU", "A100")
TIMEOUT = int(os.environ.get("TIMEOUT", 30))
CACHE_VERSION = "v1"

# Modal Volume for cache
AF2_CACHE_VOLUME = Volume.from_name("foldism-alphafold-cache", create_if_missing=True)
CACHE_DIR = "/cache"


class AlphafoldParams(TypedDict, total=False):
    """Parameters for AlphaFold inference."""
    input_str: str  # Required: FASTA content
    input_name: str  # Optional: filename (default: input.fasta)
    models: list[int]  # Optional: model numbers to run (default: [1])
    num_recycles: int  # Optional: number of recycles (default: 3)
    num_relax: int  # Optional: relaxation steps (default: 0)
    use_templates: bool  # Optional: use PDB templates (default: False)


image = (
    Image.micromamba(python_version="3.11")
    .apt_install("wget", "git")
    .pip_install(
        "colabfold[alphafold-minus-jax]@git+https://github.com/sokrypton/ColabFold@a134f6a8f8de5c41c63cb874d07e1a334cb021bb"
    )
    .micromamba_install(
        "kalign2=2.04",
        "hhsuite=3.3.0",
        "pdbfixer=1.9",
        channels=["conda-forge", "bioconda"],
    )
    .run_commands(
        'pip install --upgrade "jax[cuda12_pip]==0.5.3" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html',
        gpu="a10g",
    )
    .run_commands("python -m colabfold.download")
)

app = App("alphafold", image=image)


def _generate_cache_key(params: dict[str, Any]) -> str:
    """Generate deterministic cache key from params dict."""
    input_hash = sha256(params.get("input_str", "").encode()).hexdigest()
    cache_params = {
        "version": CACHE_VERSION,
        "input_hash": input_hash,
        "models": params.get("models", [1]),
        "num_recycles": params.get("num_recycles", 3),
        "num_relax": params.get("num_relax", 0),
        "use_templates": params.get("use_templates", False),
    }
    params_str = json.dumps(cache_params, sort_keys=True)
    return sha256(params_str.encode()).hexdigest()[:16]


@app.function(
    timeout=TIMEOUT * 60,
    gpu=GPU,
    volumes={CACHE_DIR: AF2_CACHE_VOLUME},
)
def alphafold_from_file(
    params: dict[str, Any],
    overwrite: bool = False,
) -> list:
    """Run AlphaFold2/ColabFold on input and return output files.

    Args:
        params: AlphafoldParams dict with keys:
            - input_str (required): FASTA content
            - input_name: filename (default: input.fasta)
            - models: model numbers to run (default: [1])
            - num_recycles: number of recycles (default: 3)
            - num_relax: relaxation steps (default: 0)
            - use_templates: use PDB templates (default: False)
        overwrite: If True, bypass cache and recompute

    Returns:
        List of (relative_path, bytes) tuples for output files.
    """
    import zipfile
    from io import BytesIO
    from tempfile import TemporaryDirectory

    from colabfold.batch import run as colabfold_run

    input_str = params["input_str"]
    input_name = params.get("input_name", "input.fasta")
    models = params.get("models", [1])
    num_recycles = params.get("num_recycles", 3)
    num_relax = params.get("num_relax", 0)
    use_templates = params.get("use_templates", False)

    # Generate cache key
    cache_key = _generate_cache_key(params)
    cache_path = Path(f"{CACHE_DIR}/{cache_key}")
    cache_marker = cache_path / "_COMPLETE"

    # Check cache
    if not overwrite and cache_marker.exists():
        print(f"Cache hit: {cache_key}")
        AF2_CACHE_VOLUME.reload()
        return [
            (out_file.relative_to(cache_path), out_file.read_bytes())
            for out_file in cache_path.glob("**/*")
            if out_file.is_file() and out_file.name != "_COMPLETE"
        ]

    print(f"Cache miss: {cache_key} - running inference")

    with TemporaryDirectory() as in_dir, TemporaryDirectory() as out_dir:
        fasta_path = Path(in_dir) / input_name
        fasta_path.write_text(input_str)

        colabfold_run(
            queries=fasta_path,
            result_dir=Path(out_dir),
            use_templates=use_templates,
            num_recycle=num_recycles,
            num_models=len(models),
            model_order=models,
            num_relax=num_relax,
            model_type="alphafold2_multimer_v3" if ":" in input_str else "alphafold2_ptm",
        )

        # Create a zip file of all outputs
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for out_file in Path(out_dir).rglob("*"):
                if out_file.is_file():
                    arcname = out_file.relative_to(out_dir)
                    zf.write(out_file, arcname)

        zip_bytes = zip_buffer.getvalue()
        run_name = Path(input_name).stem
        outputs = [(Path(f"{run_name}.zip"), zip_bytes)]

        # Save to cache
        cache_path.mkdir(parents=True, exist_ok=True)
        for rel_path, content in outputs:
            out_path = cache_path / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(content)

        cache_marker.touch()
        AF2_CACHE_VOLUME.commit()

        return outputs


@app.local_entrypoint()
def main(
    input_fasta: str,
    out_dir: str = "./out/alphafold",
    num_recycles: int = 3,
    overwrite: bool = False,
):
    """Run AlphaFold2 structure prediction.

    Args:
        input_fasta: Path to input FASTA file
        out_dir: Output directory
        num_recycles: Number of recycles (default: 3)
        overwrite: Bypass cache if True
    """
    input_content = open(input_fasta).read()
    run_name = Path(input_fasta).stem

    outputs = alphafold_from_file.remote(
        params={
            "input_str": input_content,
            "input_name": f"{run_name}.fasta",
            "num_recycles": num_recycles,
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
