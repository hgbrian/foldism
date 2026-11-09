# /// script
# requires-python = ">=3.12"
# dependencies = ["modal"]
# ///
"""Pinned Mol-1 backend on Modal, with worker-side confidence scoring."""

from __future__ import annotations

from pathlib import Path

from modal import Image

from .common import (
    CACHE_VOLUME,
    MODEL_VOLUME,
    TIMEOUT,
    app,
    extract_chain_sequences,
    mol1_cache_key,
)
from .opendde import _fasta_to_opendde_json

MOL1_REVISION = "e6e64866ee89640a91cef7ee777a85f1d8be03bb"
MOL1_CHECKPOINT_SHA256 = (
    "d5b83a86884aa6870565b1a249e8f13c232dfc6dfa311089ab0c7868212b6bd6"
)
MOL1_MODEL_ROOT = "/models/mol1"


def fetch_assets():
    """Cache released assets and validate the exact benchmark checkpoint."""
    import hashlib
    import json
    import shutil
    import tarfile
    import urllib.request

    destination = Path(MOL1_MODEL_ROOT)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in ("common.tar.gz", "search_database.tar.gz", "checkpoint/mol-1-base.pt"):
        url = f"https://mol-1-release.oss-cn-shanghai.aliyuncs.com/{name}"
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            partial = path.with_suffix(path.suffix + ".partial")
            with (
                urllib.request.urlopen(url, timeout=180) as source,
                partial.open("wb") as output,
            ):
                shutil.copyfileobj(source, output, length=1024 * 1024)
            partial.replace(path)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if name.endswith(".pt") and digest != MOL1_CHECKPOINT_SHA256:
            raise ValueError("Mol-1 checkpoint differs from the validated benchmark")
        manifest[name] = {"url": url, "sha256": digest, "bytes": path.stat().st_size}
        if name.endswith(".tar.gz"):
            with tarfile.open(path) as archive:
                archive.extractall(destination, filter="data")
    (destination / "asset_manifest.json").write_text(json.dumps(manifest, indent=2))
    MODEL_VOLUME.commit()


mol1_image = (
    Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "build-essential", "ninja-build", "libgl1", "libglib2.0-0")
    .env(
        {"MOL_ROOT_DIR": MOL1_MODEL_ROOT, "MAX_JOBS": "4", "PYTHONPATH": "/opt/foldism"}
    )
    .pip_install(
        "torch==2.10.0",
        f"molence[cuda12] @ git+https://github.com/molence/Mol-1.git@{MOL1_REVISION}",
        "gemmi",
        "numpy",
    )
    .run_function(fetch_assets, volumes={"/models": MODEL_VOLUME}, timeout=2400)
    .add_local_file(
        Path(__file__).with_name("confidence.py"),
        "/opt/foldism/confidence.py",
    )
)


@app.function(
    image=mol1_image,
    gpu="A100-80GB",
    cpu=8,
    memory=65536,
    timeout=TIMEOUT * 60,
    max_containers=1,
    scaledown_window=5,
    volumes={"/models": MODEL_VOLUME, "/cache": CACHE_VOLUME},
)
def mol1_predict(
    params: dict, overwrite: bool = False, job_id: str | None = None
) -> list:
    """Run one input with shared Foldism MSAs and compact confidence output."""
    import os
    import subprocess
    import sys
    from tempfile import TemporaryDirectory

    from .common import write_log_line
    from .confidence import downloadable, finalize_native_outputs

    MODEL_VOLUME.reload()
    CACHE_VOLUME.reload()
    from .confidence import cache_supports

    cache_path = Path("/cache/mol1") / mol1_cache_key(params)
    marker = cache_path / "_COMPLETE"
    if (
        marker.exists()
        and not overwrite
        and cache_supports(cache_path, bool(params.get("calculate_actifptm", True) or params.get("download_confidence_data", True)), bool(params.get("download_confidence_data", True)))
    ):
        return [
            (path.relative_to(cache_path), path.read_bytes())
            for path in cache_path.rglob("*")
            if path.is_file() and downloadable(path)
        ]
    use_msa = params.get("use_msa", True)
    msa_result = params.get("msa_result") if use_msa else None
    if use_msa and not msa_result:
        raise ValueError("Mol-1 requires shared MSAs when Use MSA is enabled")
    active = bool(params.get("calculate_actifptm", True) or params.get("download_confidence_data", True))
    with TemporaryDirectory() as directory:
        root = Path(directory)
        input_path, output = root / "input.json", root / "out"
        input_path.write_text(
            _fasta_to_opendde_json(
                params["input_str"],
                params.get("input_name", "input"),
                params.get("seed", 42),
                msa_result,
                root / "msas",
            )
        )
        command = [
            sys.executable,
            "/opt/foldism/confidence.py",
            "mol1",
            "pred",
            "--input",
            str(input_path),
            "--out_dir",
            str(output),
            "--seeds",
            str(params.get("seed", 42)),
            "--sample",
            str(params.get("num_samples", 1)),
            "--step",
            str(params.get("num_steps", 200)),
            "--cycle",
            str(params.get("num_cycles", 20)),
            "--dtype",
            "bf16",
            "--device",
            "cuda",
            "--use_msa",
            str(use_msa).lower(),
            "--use_template",
            "false",
            "--use_rna_msa",
            "false",
            "--need_atom_confidence",
            "true",
            "--load_checkpoint_path",
            MOL1_MODEL_ROOT + "/checkpoint/mol-1-base.pt",
        ]
        environment = {**os.environ, "FOLDISM_ACTIFPTM": "1" if active else "0"}
        environment["FOLDISM_CONFIDENCE_DIR"] = str(output) if params.get("download_confidence_data", True) else ""
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            write_log_line(job_id, "mol1_logs", line.rstrip(), "mol1")
        if process.wait():
            raise RuntimeError("Mol-1 inference failed; see job log")
        finalize_native_outputs(
            output, extract_chain_sequences(params["input_str"]), active
        )
        outputs = [
            (path.relative_to(output), path.read_bytes())
            for path in output.rglob("*")
            if path.is_file() and downloadable(path)
        ]
        for relative, payload in outputs:
            destination = cache_path / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        if params.get("download_confidence_data", True) and not cache_supports(cache_path, True, True):
            raise ValueError("Requested confidence arrays are missing from prediction outputs")
        marker.touch()
        CACHE_VOLUME.commit()
        return outputs
