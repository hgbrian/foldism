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
    TIMEOUT,
    alphafold_cache_key,
    app,
)

# =============================================================================
# Image
# =============================================================================

alphafold_image = (
    Image.micromamba(python_version="3.11")
    .apt_install("wget", "git")
    .pip_install(
        "colabfold[alphafold-minus-jax]@git+https://github.com/sokrypton/ColabFold@a134f6a8f8de5c41c63cb874d07e1a334cb021bb"
    )
    .micromamba_install("kalign2=2.04", "hhsuite=3.3.0", "pdbfixer=1.9", channels=["conda-forge", "bioconda"])
    .run_commands(
        'pip install --upgrade "jax[cuda12_pip]==0.5.3" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html',
        gpu="a10g",
    )
    .run_commands("python -m colabfold.download")
)

# =============================================================================
# Prediction
# =============================================================================


@app.function(
    image=alphafold_image,
    timeout=TIMEOUT * 60,
    gpu=GPU,
    volumes={"/cache": CACHE_VOLUME},
)
def alphafold_predict(params: dict[str, Any], overwrite: bool = False, job_id: str | None = None) -> list:
    """Run AlphaFold2/ColabFold structure prediction."""
    import sys
    import zipfile
    from io import BytesIO
    from tempfile import TemporaryDirectory
    from colabfold.batch import get_queries, run as colabfold_run
    from colabfold.download import default_data_dir

    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
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
            fasta_path = Path(in_dir) / input_name
            fasta_path.write_text(input_str)

            queries, is_complex = get_queries(in_dir)
            print(f"Running AF2 on {len(queries)} queries, {is_complex=}")

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
                data_dir=default_data_dir,
                zip_results=True,
            )

            run_name = Path(input_name).stem
            result_zip = Path(out_dir) / f"{run_name}.result.zip"
            if result_zip.exists():
                outputs = [(Path(f"{run_name}.zip"), result_zip.read_bytes())]
            else:
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for f in Path(out_dir).rglob("*"):
                        if f.is_file():
                            zf.write(f, f.relative_to(out_dir))
                outputs = [(Path(f"{run_name}.zip"), zip_buffer.getvalue())]

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
